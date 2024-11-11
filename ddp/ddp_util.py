import os
import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, Callable
import time
import sys

class ConfigValidationError(Exception):
    pass

class TrainingMonitor:
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.start_time = time.time()
        self.num_examples = 0

    def update(self, num_examples: int):
        self.num_examples += num_examples

    def should_log(self, batch_idx: int) -> bool:
        return batch_idx % self.log_interval == 0

    def get_stats(self) -> Dict[str, float]:
        cur_time = time.time()
        elapsed = cur_time - self.start_time
        examples_per_sec = self.num_examples / elapsed
        return {
            "examples_processed": self.num_examples,
            "elapsed_time": elapsed,
            "examples_per_second": examples_per_sec
        }

class DDPWrapper:
    def __init__(self, local_rank: int):
        self.path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_ddp = self.load_config(os.path.join(self.path_to_project_root, "ddp", "config_ddp.yaml"))
        self.world_size = self.config_ddp['distributed']['world_size']
        self.local_world_size = self.config_ddp['distributed']['nodes']['master']['local_size']
        self.node_rank = self.config_ddp['distributed']['nodes']['master']['node_rank']
        self.local_rank = local_rank
        self.global_rank = self.node_rank * self.local_world_size + local_rank
        self.device = torch.device(f"cuda:{local_rank}")

    def load_config(self, path_to_config: str) -> Dict[str, Any]:
        if not os.path.exists(path_to_config):
            raise ConfigValidationError(f"Config file not found: {path_to_config}")
        with open(path_to_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def setup(self):
        """DDP 환경 설정"""
        master_node = self.config_ddp['distributed']['nodes']['master']
        os.environ['MASTER_ADDR'] = master_node['network']['add']['ip']
        os.environ['MASTER_PORT'] = master_node['network']['tcp']['port']
        torch.distributed.init_process_group(
            backend=self.config_ddp['distributed']['backend'],
            world_size=self.world_size,
            rank=self.global_rank
        )
        torch.cuda.set_device(self.local_rank)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """모델을 DDP로 래핑"""
        model = model.to(self.device)
        return nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])

    def cleanup(self):
        """DDP 환경 정리"""
        torch.distributed.destroy_process_group()

    def prepare_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int, **kwargs) -> torch.utils.data.DataLoader:
        """데이터로더를 DDP용으로 준비"""
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )

    def run_training(self, 
                    train_fn: Callable, 
                    model: nn.Module, 
                    dataset: torch.utils.data.Dataset,
                    batch_size: int,
                    **kwargs):
        """
        외부 학습 함수를 DDP로 실행
        
        Args:
            train_fn: 학습 함수. (model, dataloader, device, **kwargs)를 인자로 받아야 함
            model: 학습할 모델
            dataset: 학습 데이터셋
            batch_size: 배치 크기
            **kwargs: train_fn에 전달할 추가 인자들
        """
        try:
            self.setup()
            ddp_model = self.prepare_model(model)
            dataloader = self.prepare_dataloader(dataset, batch_size)
            
            train_fn(
                model=ddp_model,
                dataloader=dataloader,
                device=self.device,
                **kwargs
            )
        finally:
            self.cleanup()

# 사용 예시:
def example_train_fn(model, dataloader, device, epochs=3, **kwargs):
    """예시 학습 함수"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    monitor = TrainingMonitor()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            monitor.update(len(data))
            if monitor.should_log(batch_idx):
                stats = monitor.get_stats()
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                      f"Examples/sec: {stats['examples_per_second']:.2f}")

def run_worker(rank):
    ddp_wrapper = DDPWrapper(rank)
    
    # 실제 사용시에는 아래 부분을 실제 모델과 데이터셋으로 교체
    model = nn.Sequential(nn.Linear(10, 10))  # 예시 모델
    dataset = torch.utils.data.TensorDataset(  # 예시 데이터셋
        torch.randn(100, 10),
        torch.randint(0, 10, (100,))
    )
    
    # DDP 학습 실행
    ddp_wrapper.run_training(
        train_fn=example_train_fn,
        model=model,
        dataset=dataset,
        batch_size=32,
        epochs=3  # example_train_fn에 전달될 추가 인자
    )


def main():
    # Load the config file
    path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(path_to_project_root, "ddp", "config_ddp.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        sys.exit(1)

    # Check if running with torch.distributed.launch
    if len(sys.argv) > 1:
        local_rank = int(sys.argv[1])
        run_worker(local_rank)
    else:
        # Running with torch.multiprocessing.spawn
        world_size = config['distributed']['world_size']
        if torch.cuda.is_available():
            torch.multiprocessing.spawn(run_worker, nprocs=world_size, join=True)
        else:
            print("CUDA is not available. Running in CPU mode.")
            run_worker(0)

if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    main()