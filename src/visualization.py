"""
TextVibCLIP 시각화 모듈 
Research-quality visualization for TextVibCLIP continual learning experiments
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# 한글 폰트 및 스타일 설정
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

logger = logging.getLogger(__name__)


class PaperVisualizer:
    """논문용 고품질 시각화 클래스"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 시각화 결과 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 컬러 팔레트 설정 (논문용 고품질)
        self.colors = {
            'primary': '#2E86AB',      # 블루
            'secondary': '#F24236',    # 레드  
            'accent': '#F6AE2D',       # 옐로우
            'success': '#2F9B69',      # 그린
            'warning': '#F18F01',      # 오렌지
            'neutral': '#6C757D',      # 그레이
            'background': '#F8F9FA'    # 라이트 그레이
        }
        
        # 베어링 상태별 색상 매핑 (더 강하게 구분되는 팔레트)
        self.bearing_colors = {
            'H': '#2ca02c',   # 선명한 그린
            'B': '#1f77b4',   # 선명한 블루
            'IR': '#d62728',  # 선명한 레드
            'OR': '#ff7f0e',  # 선명한 오렌지
            'M': self.colors['accent'],
            'U': '#9B59B6',
            'L': '#8E44AD'
        }

        # 베어링 상태별 마커 (모양으로도 구분)
        self.condition_markers = {
            'H': 'o',   # 원
            'B': 's',   # 사각형
            'IR': '^',  # 삼각형
            'OR': 'D',  # 다이아몬드
            'M': 'v',
            'U': 'P',
            'L': 'X'
        }
        
        # 베어링 타입별 마커
        self.bearing_markers = {
            'N204': 'o',      # 원형
            'NJ204': 's',     # 사각형
            '6204': '^',      # 삼각형
            '30204': 'D',     # 다이아몬드
            'deep_groove_ball': 'o',  # CWRU - 원형 (기본)
            'SKF': 'o'        # CWRU 대체 - 원형
        }
        
        logger.info(f"PaperVisualizer 초기화 완료: {output_dir}")
    
    def create_encoder_alignment_plot(self, 
                                    text_embeddings: np.ndarray,
                                    vib_embeddings: np.ndarray,
                                    labels: List[str],
                                    bearing_types: List[str],
                                    domain_name: str,
                                    save_name: str = "encoder_alignment") -> str:
        """
        첫 번째 도메인에서 두 encoder의 alignment 확인용 t-SNE 시각화
        """
        logger.info(f"Encoder alignment 시각화 생성 중: {domain_name}")
        
        # 안전: torch.Tensor → CPU numpy 로 변환
        def _to_numpy(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
            except Exception:
                pass
            return np.asarray(x)
        text_np = _to_numpy(text_embeddings)
        vib_np = _to_numpy(vib_embeddings)

        # 데이터 준비
        all_embeddings = np.concatenate([text_np, vib_np], axis=0)
        modality_labels = ['Text'] * len(text_np) + ['Vibration'] * len(vib_np)
        condition_labels = labels + labels  # 두 번 반복 (text + vib)
        
        # t-SNE 차원 축소
        logger.info("t-SNE 차원 축소 실행 중...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//4))
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # 서브플롯 생성 (2x2 layout) - 순서: [Text | Vibration] / [빈칸 | Modality]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Encoder Alignment Analysis - {domain_name}', fontsize=16, fontweight='bold')
        
        # 1. 베어링 상태별 분포 (Text만) - 좌상단
        ax2 = axes[0, 0]
        text_mask = np.array(modality_labels) == 'Text'
        text_coords = embeddings_2d[text_mask]
        text_conditions = np.array(condition_labels)[text_mask]
        
        for condition in np.unique(text_conditions):
            mask = text_conditions == condition
            ax2.scatter(text_coords[mask, 0], text_coords[mask, 1],
                        c=self.bearing_colors.get(condition, self.colors['neutral']),
                        alpha=0.85, s=70, label=f'{condition}',
                        marker=self.condition_markers.get(condition, 'o'),
                        edgecolors='white', linewidth=0.8)
        
        ax2.set_title('Text Embeddings - Bearing Conditions', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 2. 베어링 상태별 분포 (Vibration만) - 우상단
        ax3 = axes[0, 1]
        vib_mask = np.array(modality_labels) == 'Vibration'
        vib_coords = embeddings_2d[vib_mask]
        vib_conditions = np.array(condition_labels)[vib_mask]
        
        for condition in np.unique(vib_conditions):
            mask = vib_conditions == condition
            ax3.scatter(vib_coords[mask, 0], vib_coords[mask, 1],
                        c=self.bearing_colors.get(condition, self.colors['neutral']),
                        alpha=0.85, s=70, label=f'{condition}',
                        marker=self.condition_markers.get(condition, 'o'),
                        edgecolors='white', linewidth=0.8)
        
        ax3.set_title('Vibration Embeddings - Bearing Conditions', fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 3. 모달리티별 분포 (Text vs Vibration) - 우하단
        axes[1, 0].axis('off')
        ax4 = axes[1, 1]
        for i, modality in enumerate(['Text', 'Vibration']):
            mask = np.array(modality_labels) == modality
            alpha = 0.7 if modality == 'Text' else 0.9
            marker = 'o' if modality == 'Text' else 's'
            color = self.colors['primary'] if modality == 'Text' else self.colors['secondary']
            ax4.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=color, alpha=alpha, s=60, marker=marker,
                        label=modality, edgecolors='white', linewidth=0.5)
        ax4.set_title('Modality Distribution', fontweight='bold')
        ax4.legend(frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        
        # 레이아웃 조정 및 저장
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{domain_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Encoder alignment 시각화 저장 완료: {save_path}")
        return save_path
    
    def create_continual_learning_performance_plot(self,
                                                 domain_names: List[str],
                                                 accuracies: List[float],
                                                 forgetting_scores: List[float],
                                                 scenario_name: str,
                                                 save_name: str = "continual_performance") -> str:
        """
        도메인별 정확도와 forgetting 점수 막대 그래프
        """
        logger.info(f"Continual learning 성능 시각화 생성 중: {scenario_name}")
        
        # 입력 길이 정합성 보정
        try:
            n = len(domain_names)
            acc = list(accuracies) if accuracies is not None else []
            fog = list(forgetting_scores) if forgetting_scores is not None else []
            # 길이 초과 시 자르기
            acc = acc[:n]
            fog = fog[:n]
            # 길이 부족 시 0으로 패딩
            if len(acc) < n:
                acc = acc + [0.0] * (n - len(acc))
            if len(fog) < n:
                fog = fog + [0.0] * (n - len(fog))
        except Exception:
            # 실패 시 보수적으로 도메인 수만큼 0 배열 사용
            n = len(domain_names)
            acc = [0.0] * n
            fog = [0.0] * n

        # Figure 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Continual Learning Performance - {scenario_name}', 
                    fontsize=16, fontweight='bold')
        
        x_positions = np.arange(len(domain_names))
        bar_width = 0.6
        
        # 1. 정확도 막대 그래프
        bars1 = ax1.bar(x_positions, acc, bar_width, 
                        color=self.colors['primary'], alpha=0.8, 
                        edgecolor='white', linewidth=1.5)
        
        # 정확도 값 표시
        for i, (bar, a) in enumerate(zip(bars1, acc)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{a:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Accuracy per Domain', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(domain_names)
        ax1.set_ylim(0, (max(acc) if len(acc) else 1.0) * 1.15 if any(np.isfinite(acc)) else 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 80% 기준선 추가
        ax1.axhline(y=0.8, color=self.colors['success'], linestyle='--', 
                   linewidth=2, alpha=0.7, label='80% Target')
        ax1.legend()
        
        # 2. Forgetting 점수 막대 그래프
        bars2 = ax2.bar(x_positions, fog, bar_width,
                        color=self.colors['secondary'], alpha=0.8,
                        edgecolor='white', linewidth=1.5)
        
        # Forgetting 값 표시
        for i, (bar, forget) in enumerate(zip(bars2, fog)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{forget:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Forgetting Score per Domain', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Forgetting Score', fontweight='bold')
        ax2.set_xlabel('Domains', fontweight='bold')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(domain_names)
        ax2.set_ylim(0, (max(fog) * 1.15) if len(fog) else 0.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 레이아웃 조정 및 저장
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{scenario_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Continual learning 성능 시각화 저장 완료: {save_path}")
        return save_path
    
    def create_domain_shift_robustness_plot(self,
                                          domain_embeddings: Dict[str, Dict[str, np.ndarray]],
                                          scenario_name: str,
                                          save_name: str = "domain_shift_robustness") -> str:
        """
        도메인 시프트와 continual learning robustness 시각화
        """
        logger.info(f"Domain shift robustness 시각화 생성 중: {scenario_name}")
        
        # 모든 도메인의 임베딩을 결합
        all_text_embeddings = []
        all_vib_embeddings = []
        domain_labels = []
        
        for domain, embeddings in domain_embeddings.items():
            if 'text' in embeddings and 'vib' in embeddings:
                def _to_numpy(x):
                    try:
                        import torch
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu().numpy()
                    except Exception:
                        pass
                    return np.asarray(x)
                text_emb = _to_numpy(embeddings['text'])
                vib_emb = _to_numpy(embeddings['vib'])
                
                # 샘플링 (시각화를 위해 각 도메인에서 최대 100개)
                n_samples = min(100, len(text_emb))
                indices = np.random.choice(len(text_emb), n_samples, replace=False)
                
                all_text_embeddings.append(text_emb[indices])
                all_vib_embeddings.append(vib_emb[indices])
                domain_labels.extend([domain] * n_samples)
        
        if not all_text_embeddings:
            logger.warning("도메인 임베딩 데이터가 없습니다.")
            return ""
        
        # 임베딩 결합
        text_embeddings = np.vstack(all_text_embeddings)
        vib_embeddings = np.vstack(all_vib_embeddings)
        all_embeddings = np.vstack([text_embeddings, vib_embeddings])
        
        # 모달리티 라벨
        modality_labels = ['Text'] * len(text_embeddings) + ['Vibration'] * len(vib_embeddings)
        domain_labels_full = domain_labels + domain_labels
        
        # PCA 차원 축소 (더 안정적)
        logger.info("PCA 차원 축소 실행 중...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        # Figure 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Domain Shift Analysis - {scenario_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 도메인별 분포 (Text 임베딩)
        text_mask = np.array(modality_labels) == 'Text'
        text_coords = embeddings_2d[text_mask]
        text_domains = np.array(domain_labels_full)[text_mask]
        
        unique_domains = np.unique(domain_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
        
        for i, domain in enumerate(unique_domains):
            mask = text_domains == domain
            ax1.scatter(text_coords[mask, 0], text_coords[mask, 1],
                       c=[colors[i]], alpha=0.7, s=60, label=f'{domain}',
                       edgecolors='white', linewidth=0.8)
        
        ax1.set_title('Text Embeddings - Domain Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 도메인별 분포 (Vibration 임베딩)
        vib_mask = np.array(modality_labels) == 'Vibration'
        vib_coords = embeddings_2d[vib_mask]
        vib_domains = np.array(domain_labels_full)[vib_mask]
        
        for i, domain in enumerate(unique_domains):
            mask = vib_domains == domain
            ax2.scatter(vib_coords[mask, 0], vib_coords[mask, 1],
                       c=[colors[i]], alpha=0.7, s=60, label=f'{domain}',
                       edgecolors='white', linewidth=0.8)
        
        ax2.set_title('Vibration Embeddings - Domain Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PCA 설명력 추가
        explained_variance = pca.explained_variance_ratio_
        fig.text(0.5, 0.02, f'PCA Explained Variance: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}', 
                ha='center', fontsize=10, style='italic')
        
        # 레이아웃 조정 및 저장
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{scenario_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Domain shift robustness 시각화 저장 완료: {save_path}")
        return save_path

    def create_similarity_diagnostics_plot(self,
                                           vib_embeddings: Any,
                                           labels: List[str],
                                           prompt_embeddings: Any,
                                           domain_name: str,
                                           save_name: str = "similarity_diagnostics") -> str:
        """진동 임베딩과 클래스 프로토타입 간 유사도/마진 분포 시각화
        Args:
            vib_embeddings: (N, D) torch.Tensor 또는 np.ndarray
            labels: 길이 N의 클래스 라벨 리스트 (예: ['H','B','IR','OR'])
            prompt_embeddings: (C, D) torch.Tensor 또는 np.ndarray, 클래스 프로토타입
            domain_name: '0HP' 등
        Returns:
            저장 경로
        """
        logger.info(f"Similarity diagnostics 시각화 생성 중: {domain_name}")

        # to torch tensor (CPU)
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return torch.tensor(np.asarray(x), dtype=torch.float32)

        vib_t = _to_tensor(vib_embeddings)
        proto_t = _to_tensor(prompt_embeddings)

        # 라벨 매핑
        label_map = {'H': 0, 'B': 1, 'IR': 2, 'OR': 3}
        y = torch.tensor([label_map.get(l, 0) for l in labels], dtype=torch.long)

        # 정규화 및 유사도
        vib_t = F.normalize(vib_t, p=2, dim=1)
        proto_t = F.normalize(proto_t, p=2, dim=1)
        sims = vib_t @ proto_t.T  # (N, C)

        # 정답/오답 유사도, 마진
        correct_sim = sims[torch.arange(sims.size(0)), y]
        sims_masked = sims.clone()
        sims_masked[torch.arange(sims.size(0)), y] = -1e9
        best_incorrect, _ = sims_masked.max(dim=1)
        margins = correct_sim - best_incorrect

        # per-class 데이터 준비
        classes = ['H', 'B', 'IR', 'OR']
        data = {cls: {'correct': [], 'margin': []} for cls in classes}
        for i, cls_idx in enumerate(y.tolist()):
            cls = classes[cls_idx]
            data[cls]['correct'].append(float(correct_sim[i].item()))
            data[cls]['margin'].append(float(margins[i].item()))

        # Figure: 2x1 (상: 정답 유사도 상자/바이올린, 하: 마진 히스토그램)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Similarity Diagnostics - {domain_name}', fontsize=16, fontweight='bold')

        # 상단: per-class 정답 유사도 박스+스트립
        cls_order = classes
        box_data = [data[c]['correct'] for c in cls_order]
        bplot = ax1.boxplot(box_data, labels=cls_order, patch_artist=True)
        colors = [self.bearing_colors.get(c, self.colors['neutral']) for c in cls_order]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        # 점 산포
        for i, c in enumerate(cls_order):
            jitter_x = np.random.normal(loc=i+1, scale=0.05, size=len(data[c]['correct']))
            ax1.scatter(jitter_x, data[c]['correct'], c=colors[i], s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax1.set_ylabel('Cosine similarity to correct prototype')
        ax1.grid(True, alpha=0.3)

        # 하단: per-class margin 히스토그램
        for i, c in enumerate(cls_order):
            ax2.hist(data[c]['margin'], bins=20, alpha=0.6, label=f'{c}', color=colors[i])
        ax2.set_xlabel('Margin (correct - best incorrect)')
        ax2.set_ylabel('Count')
        ax2.axvline(0.0, color=self.colors['secondary'], linestyle='--', linewidth=1.5, label='0 margin')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 저장
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{domain_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Similarity diagnostics 시각화 저장 완료: {save_path}")
        return save_path


def create_visualizer(output_dir: str) -> PaperVisualizer:
    """PaperVisualizer 인스턴스 생성 (기존 호환성 유지)"""
    return PaperVisualizer(output_dir)
