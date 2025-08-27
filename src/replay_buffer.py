"""
Replay Buffer: Continual Learning을 위한 메모리 버퍼
이전 도메인 임베딩 저장 및 샘플링
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import random

from configs.model_config import TRAINING_CONFIG

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    임베딩 기반 Replay Buffer
    
    각 도메인별로 representative embeddings을 저장하고
    새 도메인 학습 시 이전 임베딩들을 샘플링하여 catastrophic forgetting 방지
    """
    
    def __init__(self, 
                 buffer_size_per_domain: int = TRAINING_CONFIG['replay_buffer_size'],
                 embedding_dim: int = 512,
                 sampling_strategy: str = 'random'):
        """
        Args:
            buffer_size_per_domain (int): 도메인당 저장할 임베딩 수
            embedding_dim (int): 임베딩 차원
            sampling_strategy (str): 샘플링 전략 ('random', 'balanced')
        """
        self.buffer_size_per_domain = buffer_size_per_domain
        self.embedding_dim = embedding_dim
        self.sampling_strategy = sampling_strategy
        
        # 도메인별 저장소
        self.text_embeddings = {}  # {domain_id: tensor}
        self.vib_embeddings = {}   # {domain_id: tensor}
        self.metadata_list = {}    # {domain_id: list of metadata}
        
        # 도메인 정보
        self.domains = []  # 순서대로 저장된 도메인 리스트
        self.current_domain = None
        
        logger.info(f"ReplayBuffer 초기화: "
                   f"buffer_size={buffer_size_per_domain}, "
                   f"embedding_dim={embedding_dim}, "
                   f"sampling={sampling_strategy}")
    
    def add_domain_data(self, 
                       domain_id: int,
                       text_embeddings: torch.Tensor,
                       vib_embeddings: torch.Tensor,
                       metadata_list: List[Dict],
                       selection_strategy: str = 'random') -> None:
        """
        새 도메인 데이터를 버퍼에 추가
        
        Args:
            domain_id (int): 도메인 ID (예: RPM 값)
            text_embeddings (torch.Tensor): 텍스트 임베딩 (N, embedding_dim)
            vib_embeddings (torch.Tensor): 진동 임베딩 (N, embedding_dim)  
            metadata_list (List[Dict]): 메타데이터 리스트
            selection_strategy (str): 선택 전략 ('random', 'diverse', 'representative')
        """
        num_samples = text_embeddings.size(0)
        
        if num_samples == 0:
            logger.warning(f"Domain {domain_id}: 빈 데이터, 건너뜀")
            return
        
        # 버퍼 크기만큼 선택
        if num_samples <= self.buffer_size_per_domain:
            # 데이터가 적으면 모두 저장
            selected_indices = list(range(num_samples))
        else:
            # 선택 전략에 따라 샘플링
            selected_indices = self._select_representative_samples(
                text_embeddings, vib_embeddings, metadata_list, selection_strategy
            )
        
        # 선택된 데이터 저장
        selected_text = text_embeddings[selected_indices]
        selected_vib = vib_embeddings[selected_indices]
        selected_metadata = [metadata_list[i] for i in selected_indices]
        
        self.text_embeddings[domain_id] = selected_text.detach().cpu()
        self.vib_embeddings[domain_id] = selected_vib.detach().cpu()
        self.metadata_list[domain_id] = selected_metadata
        
        # 도메인 목록 업데이트
        if domain_id not in self.domains:
            self.domains.append(domain_id)
        
        self.current_domain = domain_id
        
        logger.info(f"Domain {domain_id} 데이터 추가: "
                   f"{len(selected_indices)}개 샘플 저장 "
                   f"(전체 {num_samples}개 중)")
    
    def _select_representative_samples(self,
                                     text_embeddings: torch.Tensor,
                                     vib_embeddings: torch.Tensor,
                                     metadata_list: List[Dict],
                                     strategy: str) -> List[int]:
        """
        Representative samples 선택
        
        Args:
            text_embeddings: 텍스트 임베딩
            vib_embeddings: 진동 임베딩  
            metadata_list: 메타데이터
            strategy: 선택 전략
            
        Returns:
            List[int]: 선택된 인덱스 리스트
        """
        num_samples = text_embeddings.size(0)
        
        if strategy == 'random':
            # 랜덤 선택
            indices = random.sample(range(num_samples), self.buffer_size_per_domain)
            
        elif strategy == 'diverse':
            # K-means 클러스터링으로 다양성 확보
            indices = self._kmeans_selection(vib_embeddings)
            
        elif strategy == 'representative':
            # 클래스별 균등 선택
            indices = self._balanced_class_selection(metadata_list)
            
        else:
            raise ValueError(f"알 수 없는 선택 전략: {strategy}")
        
        return indices
    
    def _kmeans_selection(self, embeddings: torch.Tensor) -> List[int]:
        """K-means 클러스터링으로 다양한 샘플 선택"""
        try:
            from sklearn.cluster import KMeans
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=self.buffer_size_per_domain, random_state=42)
            embeddings_np = embeddings.detach().cpu().numpy()
            clusters = kmeans.fit_predict(embeddings_np)
            
            # 각 클러스터에서 중심에 가장 가까운 샘플 선택
            selected_indices = []
            for cluster_id in range(self.buffer_size_per_domain):
                cluster_mask = clusters == cluster_id
                if not cluster_mask.any():
                    continue
                
                cluster_indices = np.where(cluster_mask)[0]
                cluster_embeddings = embeddings_np[cluster_indices]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # 중심에 가장 가까운 샘플 찾기
                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
            
            return selected_indices
            
        except ImportError:
            logger.warning("scikit-learn 없음, 랜덤 선택으로 대체")
            return random.sample(range(embeddings.size(0)), self.buffer_size_per_domain)
    
    def _balanced_class_selection(self, metadata_list: List[Dict]) -> List[int]:
        """클래스별 균등한 샘플 선택"""
        # 클래스별 인덱스 그룹화
        class_groups = defaultdict(list)
        for i, metadata in enumerate(metadata_list):
            # 회전체상태_베어링상태_베어링타입으로 클래스 정의
            class_key = f"{metadata['rotating_component']}_{metadata['bearing_condition']}_{metadata['bearing_type']}"
            class_groups[class_key].append(i)
        
        # 클래스별 균등 선택
        selected_indices = []
        samples_per_class = max(1, self.buffer_size_per_domain // len(class_groups))
        
        for class_key, indices in class_groups.items():
            num_select = min(samples_per_class, len(indices))
            selected = random.sample(indices, num_select)
            selected_indices.extend(selected)
        
        # 부족하면 랜덤 추가
        if len(selected_indices) < self.buffer_size_per_domain:
            remaining = self.buffer_size_per_domain - len(selected_indices)
            all_indices = set(range(len(metadata_list)))
            available = list(all_indices - set(selected_indices))
            
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                selected_indices.extend(additional)
        
        return selected_indices[:self.buffer_size_per_domain]
    
    def sample_replay_data(self, 
                          num_samples: int,
                          exclude_current: bool = True,
                          device: torch.device = torch.device('cpu')) -> Optional[Dict[str, torch.Tensor]]:
        """
        Replay 데이터 샘플링
        
        Args:
            num_samples (int): 샘플링할 데이터 수
            exclude_current (bool): 현재 도메인 제외 여부
            device (torch.device): 타겟 디바이스
            
        Returns:
            Dict with sampled embeddings or None if no data
        """
        available_domains = self.domains.copy()
        
        if exclude_current and self.current_domain in available_domains:
            available_domains.remove(self.current_domain)
        
        if not available_domains:
            # 첫 도메인 학습 시에는 정상적으로 replay할 도메인이 없음
            if len(self.domains) == 0:
                logger.debug("첫 도메인 학습 - replay할 도메인이 없음 (정상)")
            else:
                # domains가 dict인지 list인지 확인 후 처리
                if isinstance(self.domains, dict):
                    domain_keys = list(self.domains.keys())
                else:
                    domain_keys = self.domains
                logger.debug(f"Replay할 도메인이 없음 - 사용 가능: {domain_keys}")
            return None
        
        # 도메인별 샘플 수 결정
        if self.sampling_strategy == 'balanced':
            # 도메인별 균등 샘플링
            samples_per_domain = max(1, num_samples // len(available_domains))
            domain_samples = {domain: samples_per_domain for domain in available_domains}
            
            # 나머지 샘플을 랜덤 도메인에 배분
            remaining = num_samples - sum(domain_samples.values())
            for _ in range(remaining):
                domain = random.choice(available_domains)
                domain_samples[domain] += 1
        else:
            # 랜덤 샘플링
            domain_samples = defaultdict(int)
            for _ in range(num_samples):
                domain = random.choice(available_domains)
                domain_samples[domain] += 1
        
        # 실제 샘플링
        sampled_text = []
        sampled_vib = []
        sampled_metadata = []
        
        for domain, num_domain_samples in domain_samples.items():
            if domain not in self.text_embeddings:
                continue
            
            domain_text = self.text_embeddings[domain]
            domain_vib = self.vib_embeddings[domain]
            domain_metadata = self.metadata_list[domain]
            
            # 해당 도메인에서 랜덤 샘플링
            available_count = domain_text.size(0)
            actual_samples = min(num_domain_samples, available_count)
            
            if actual_samples > 0:
                indices = random.sample(range(available_count), actual_samples)
                
                sampled_text.append(domain_text[indices])
                sampled_vib.append(domain_vib[indices])
                sampled_metadata.extend([domain_metadata[i] for i in indices])
        
        if not sampled_text:
            logger.warning("샘플링된 데이터가 없음")
            return None
        
        # 텐서 결합 및 디바이스 이동
        combined_text = torch.cat(sampled_text, dim=0).to(device)
        combined_vib = torch.cat(sampled_vib, dim=0).to(device)
        
        return {
            'text_embeddings': combined_text,
            'vib_embeddings': combined_vib,
            'metadata': sampled_metadata
        }
    
    def get_buffer_info(self) -> Dict:
        """버퍼 상태 정보 반환"""
        info = {
            'total_domains': len(self.domains),
            'domains': self.domains.copy(),
            'current_domain': self.current_domain,
            'domain_sizes': {}
        }
        
        for domain in self.domains:
            if domain in self.text_embeddings:
                info['domain_sizes'][domain] = self.text_embeddings[domain].size(0)
        
        return info
    
    def clear_domain(self, domain_id: int):
        """특정 도메인 데이터 삭제"""
        if domain_id in self.text_embeddings:
            del self.text_embeddings[domain_id]
            del self.vib_embeddings[domain_id]
            del self.metadata_list[domain_id]
            
            if domain_id in self.domains:
                self.domains.remove(domain_id)
            
            logger.info(f"Domain {domain_id} 데이터 삭제됨")
    
    def clear_all(self):
        """모든 데이터 삭제"""
        self.text_embeddings.clear()
        self.vib_embeddings.clear()
        self.metadata_list.clear()
        self.domains.clear()
        self.current_domain = None
        
        logger.info("모든 replay 데이터 삭제됨")
    
    def save_buffer(self, path: str):
        """버퍼 저장"""
        buffer_data = {
            'text_embeddings': self.text_embeddings,
            'vib_embeddings': self.vib_embeddings,
            'metadata_list': self.metadata_list,
            'domains': self.domains,
            'current_domain': self.current_domain,
            'config': {
                'buffer_size_per_domain': self.buffer_size_per_domain,
                'embedding_dim': self.embedding_dim,
                'sampling_strategy': self.sampling_strategy
            }
        }
        
        torch.save(buffer_data, path)
        logger.info(f"Replay buffer 저장됨: {path}")
    
    def load_buffer(self, path: str):
        """버퍼 로딩"""
        buffer_data = torch.load(path, map_location='cpu')
        
        self.text_embeddings = buffer_data['text_embeddings']
        self.vib_embeddings = buffer_data['vib_embeddings']
        self.metadata_list = buffer_data['metadata_list']
        self.domains = buffer_data['domains']
        self.current_domain = buffer_data['current_domain']
        
        # 설정 복원
        config = buffer_data.get('config', {})
        self.buffer_size_per_domain = config.get('buffer_size_per_domain', self.buffer_size_per_domain)
        self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
        self.sampling_strategy = config.get('sampling_strategy', self.sampling_strategy)
        
        logger.info(f"Replay buffer 로딩됨: {path}")


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== ReplayBuffer 테스트 ===")
    
    # 버퍼 생성
    buffer = ReplayBuffer(buffer_size_per_domain=10, embedding_dim=512)
    
    # 더미 데이터 생성
    for domain_id in [600, 800, 1000]:
        num_samples = 50
        text_emb = torch.randn(num_samples, 512)
        vib_emb = torch.randn(num_samples, 512)
        metadata = [
            {
                'rotating_component': 'H',
                'bearing_condition': 'B' if i % 2 == 0 else 'H',
                'bearing_type': '6204',
                'rotating_speed': domain_id
            }
            for i in range(num_samples)
        ]
        
        # 도메인 데이터 추가
        buffer.add_domain_data(domain_id, text_emb, vib_emb, metadata)
    
    # 버퍼 정보 확인
    info = buffer.get_buffer_info()
    print(f"버퍼 정보: {info}")
    
    # Replay 데이터 샘플링 테스트
    replay_data = buffer.sample_replay_data(20)
    if replay_data:
        print(f"Replay 텍스트 임베딩 shape: {replay_data['text_embeddings'].shape}")
        print(f"Replay 진동 임베딩 shape: {replay_data['vib_embeddings'].shape}")
        print(f"Replay 메타데이터 수: {len(replay_data['metadata'])}")
    
    print("\n=== ReplayBuffer 테스트 완료 ===")
