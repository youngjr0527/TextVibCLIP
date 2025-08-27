"""
TextVibCLIP 고급 시각화 모듈
논문 품질의 Figure 생성을 위한 시각화 함수들
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings

# Warning 억제
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except:
    pass

# 선택적 import (설치되지 않은 경우 대체 기능 제공)
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️ seaborn이 설치되지 않았습니다. 일부 시각화가 제한됩니다.")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn이 설치되지 않았습니다. t-SNE 시각화가 제한됩니다.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# 논문 품질 설정
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'figure.dpi': 300
})

# 색맹 친화적 색상 팔레트
COLORBLIND_PALETTE = [
    '#1f77b4',  # 파랑
    '#ff7f0e',  # 주황  
    '#2ca02c',  # 초록
    '#d62728',  # 빨강
    '#9467bd',  # 보라
    '#8c564b',  # 갈색
    '#e377c2',  # 분홍
    '#7f7f7f',  # 회색
    '#bcbd22',  # 올리브
    '#17becf'   # 청록
]


class AdvancedVisualizer:
    """논문용 고급 시각화 클래스"""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Args:
            output_dir (str): 시각화 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"AdvancedVisualizer 초기화: {output_dir}")
    
    def plot_advanced_tsne(self, 
                          domain_results: Dict[int, Dict],
                          scenario_name: str = "Scenario",
                          max_samples_per_domain: int = 200,
                          save_path: Optional[str] = None) -> str:
        """
        라벨별 구분이 있는 고급 t-SNE 시각화
        
        Args:
            domain_results: 도메인별 임베딩 결과
            scenario_name: 시나리오 이름
            max_samples_per_domain: 도메인당 최대 샘플 수
            save_path: 저장 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn이 설치되지 않아 t-SNE를 실행할 수 없습니다.")
            return None
            
        try:
            # 임베딩과 라벨 수집
            all_embeddings = []
            all_labels = []
            all_modalities = []
            all_domains = []
            
            for domain_id, results in domain_results.items():
                text_emb = results['text_embeddings']
                vib_emb = results['vib_embeddings']
                metadata = results['metadata']
                
                # 샘플링
                num_samples = min(max_samples_per_domain, len(text_emb))
                indices = torch.randperm(len(text_emb))[:num_samples]
                
                # 텍스트 임베딩
                all_embeddings.append(text_emb[indices])
                all_modalities.extend(['Text'] * num_samples)
                all_domains.extend([f'D{domain_id}'] * num_samples)
                
                # 진동 임베딩
                all_embeddings.append(vib_emb[indices])
                all_modalities.extend(['Vibration'] * num_samples)
                all_domains.extend([f'D{domain_id}'] * num_samples)
                
                # 라벨 추출 (메타데이터에서)
                for i in indices:
                    meta = metadata[i] if i < len(metadata) else metadata[0]
                    bearing_condition = meta.get('bearing_condition', 'Unknown')
                    all_labels.extend([bearing_condition, bearing_condition])  # 텍스트, 진동 동일 라벨
            
            # 임베딩 결합 (CUDA → CPU 변환)
            embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
            
            # t-SNE 실행
            logger.info("고급 t-SNE 실행 중...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 서브플롯 1: 모달리티별 구분
            unique_modalities = list(set(all_modalities))
            for i, modality in enumerate(unique_modalities):
                indices = [j for j, m in enumerate(all_modalities) if m == modality]
                x = embeddings_2d[indices, 0]
                y = embeddings_2d[indices, 1]
                
                marker = 'o' if modality == 'Text' else '^'
                ax1.scatter(x, y, c=COLORBLIND_PALETTE[i], label=modality, 
                           marker=marker, alpha=0.7, s=50)
            
            ax1.set_xlabel('t-SNE Component 1')
            ax1.set_ylabel('t-SNE Component 2')
            ax1.set_title(f'{scenario_name}: Embedding Space by Modality')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 서브플롯 2: 라벨별 구분
            unique_labels = list(set(all_labels))
            for i, label in enumerate(unique_labels):
                indices = [j for j, l in enumerate(all_labels) if l == label]
                x = embeddings_2d[indices, 0]
                y = embeddings_2d[indices, 1]
                
                # 모달리티별 마커
                text_indices = [j for j in indices if all_modalities[j] == 'Text']
                vib_indices = [j for j in indices if all_modalities[j] == 'Vibration']
                
                if text_indices:
                    ax2.scatter(embeddings_2d[text_indices, 0], embeddings_2d[text_indices, 1], 
                               c=COLORBLIND_PALETTE[i], marker='o', label=f'{label}-Text', 
                               alpha=0.7, s=50)
                if vib_indices:
                    ax2.scatter(embeddings_2d[vib_indices, 0], embeddings_2d[vib_indices, 1], 
                               c=COLORBLIND_PALETTE[i], marker='^', label=f'{label}-Vib', 
                               alpha=0.7, s=50)
            
            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')
            ax2.set_title(f'{scenario_name}: Embedding Space by Fault Type')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = self.output_dir / f'{scenario_name}_advanced_tsne.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"고급 t-SNE 저장됨: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"고급 t-SNE 생성 실패: {e}")
            return None
    
    def plot_confusion_matrices(self,
                               domain_results: Dict[int, Dict],
                               scenario_name: str = "Scenario",
                               save_path: Optional[str] = None) -> str:
        """
        도메인별 Confusion Matrix 시각화
        
        Args:
            domain_results: 도메인별 결과
            scenario_name: 시나리오 이름
            save_path: 저장 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            num_domains = len(domain_results)
            cols = min(3, num_domains)
            rows = (num_domains + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if num_domains == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (domain_id, results) in enumerate(domain_results.items()):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                # 실제 라벨과 예측 라벨 생성 (간단한 threshold 기반)
                text_emb = results['text_embeddings']
                vib_emb = results['vib_embeddings']
                metadata = results['metadata']
                
                # 유사도 계산 (CUDA → CPU 변환)
                similarities = torch.cosine_similarity(text_emb, vib_emb, dim=1)
                predictions = (similarities > 0.5).long().cpu().numpy()
                targets = np.ones_like(predictions)  # 모든 쌍이 positive
                
                # Confusion Matrix 생성 (Binary classification)
                cm = confusion_matrix(targets, predictions)
                
                # 히트맵 그리기
                if SEABORN_AVAILABLE:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Negative', 'Positive'],
                               yticklabels=['Negative', 'Positive'])
                else:
                    # seaborn 없으면 matplotlib으로 대체
                    im = ax.imshow(cm, cmap='Blues')
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(['Negative', 'Positive'])
                    ax.set_yticklabels(['Negative', 'Positive'])
                    
                    # 값 표시
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                
                domain_name = f'Domain {domain_id}'
                accuracy = (predictions == targets).mean()
                ax.set_title(f'{domain_name} (Acc: {accuracy:.3f})')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
            
            # 빈 서브플롯 숨기기
            for idx in range(num_domains, rows * cols):
                row = idx // cols
                col = idx % cols
                if rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
            
            plt.suptitle(f'{scenario_name}: Confusion Matrices', fontsize=16)
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = self.output_dir / f'{scenario_name}_confusion_matrices.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrices 저장됨: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Confusion matrix 생성 실패: {e}")
            return None
    
    def plot_similarity_heatmap(self,
                               text_embeddings: torch.Tensor,
                               vib_embeddings: torch.Tensor,
                               text_labels: List[str],
                               vib_labels: List[str],
                               scenario_name: str = "Scenario",
                               max_samples: int = 50,
                               save_path: Optional[str] = None) -> str:
        """
        Cross-modal Similarity Heatmap 생성
        
        Args:
            text_embeddings: 텍스트 임베딩 (N, 512)
            vib_embeddings: 진동 임베딩 (N, 512)
            text_labels: 텍스트 라벨 리스트
            vib_labels: 진동 라벨 리스트
            scenario_name: 시나리오 이름
            max_samples: 최대 샘플 수 (계산 효율성)
            save_path: 저장 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 샘플링 (너무 크면 히트맵이 복잡함)
            num_samples = min(max_samples, len(text_embeddings))
            indices = torch.randperm(len(text_embeddings))[:num_samples]
            
            text_emb_sample = text_embeddings[indices]
            vib_emb_sample = vib_embeddings[indices]
            text_labels_sample = [text_labels[i] for i in indices]
            vib_labels_sample = [vib_labels[i] for i in indices]
            
            # L2 정규화
            text_emb_norm = torch.nn.functional.normalize(text_emb_sample, dim=1)
            vib_emb_norm = torch.nn.functional.normalize(vib_emb_sample, dim=1)
            
            # 유사도 행렬 계산 (CUDA → CPU 변환)
            similarity_matrix = torch.matmul(text_emb_norm, vib_emb_norm.t()).cpu().numpy()
            
            # 시각화
            plt.figure(figsize=(12, 10))
            
            # 히트맵 생성
            sns.heatmap(similarity_matrix, 
                       xticklabels=vib_labels_sample,
                       yticklabels=text_labels_sample,
                       cmap='RdYlBu_r',
                       center=0,
                       annot=False,
                       cbar_kws={'label': 'Cosine Similarity'})
            
            plt.title(f'{scenario_name}: Cross-Modal Similarity Matrix')
            plt.xlabel('Vibration Embeddings')
            plt.ylabel('Text Embeddings')
            
            # 대각선 강조 (positive pairs)
            for i in range(min(len(text_labels_sample), len(vib_labels_sample))):
                plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                                 edgecolor='red', linewidth=2))
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = self.output_dir / f'{scenario_name}_similarity_heatmap.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Similarity heatmap 저장됨: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Similarity heatmap 생성 실패: {e}")
            return None
    
    def plot_domain_shift_analysis(self,
                                  domain_results: Dict[int, Dict],
                                  scenario_name: str = "Scenario",
                                  save_path: Optional[str] = None) -> str:
        """
        Domain Shift 분석 시각화
        
        Args:
            domain_results: 도메인별 결과
            scenario_name: 시나리오 이름
            save_path: 저장 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 도메인별 임베딩 중심 계산
            domain_centers = {}
            domain_names = []
            
            for domain_id, results in domain_results.items():
                text_emb = results['text_embeddings']
                vib_emb = results['vib_embeddings']
                
                # 각 모달리티별 중심 계산
                text_center = torch.mean(text_emb, dim=0)
                vib_center = torch.mean(vib_emb, dim=0)
                
                domain_centers[domain_id] = {
                    'text_center': text_center,
                    'vib_center': vib_center,
                    'combined_center': (text_center + vib_center) / 2
                }
                domain_names.append(str(domain_id))
            
            # 도메인간 거리 계산
            domain_ids = list(domain_centers.keys())
            num_domains = len(domain_ids)
            
            text_distances = np.zeros((num_domains, num_domains))
            vib_distances = np.zeros((num_domains, num_domains))
            combined_distances = np.zeros((num_domains, num_domains))
            
            for i, domain1 in enumerate(domain_ids):
                for j, domain2 in enumerate(domain_ids):
                    # 코사인 거리 계산
                    text_dist = 1 - torch.cosine_similarity(
                        domain_centers[domain1]['text_center'].unsqueeze(0),
                        domain_centers[domain2]['text_center'].unsqueeze(0)
                    ).item()
                    
                    vib_dist = 1 - torch.cosine_similarity(
                        domain_centers[domain1]['vib_center'].unsqueeze(0),
                        domain_centers[domain2]['vib_center'].unsqueeze(0)
                    ).item()
                    
                    combined_dist = 1 - torch.cosine_similarity(
                        domain_centers[domain1]['combined_center'].unsqueeze(0),
                        domain_centers[domain2]['combined_center'].unsqueeze(0)
                    ).item()
                    
                    text_distances[i, j] = text_dist
                    vib_distances[i, j] = vib_dist
                    combined_distances[i, j] = combined_dist
            
            # 시각화
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 텍스트 임베딩 거리
            sns.heatmap(text_distances, annot=True, fmt='.3f', cmap='viridis',
                       xticklabels=domain_names, yticklabels=domain_names, ax=ax1)
            ax1.set_title('Text Embedding Distances')
            
            # 2. 진동 임베딩 거리
            sns.heatmap(vib_distances, annot=True, fmt='.3f', cmap='viridis',
                       xticklabels=domain_names, yticklabels=domain_names, ax=ax2)
            ax2.set_title('Vibration Embedding Distances')
            
            # 3. 결합 임베딩 거리
            sns.heatmap(combined_distances, annot=True, fmt='.3f', cmap='viridis',
                       xticklabels=domain_names, yticklabels=domain_names, ax=ax3)
            ax3.set_title('Combined Embedding Distances')
            
            # 4. 도메인 순서별 거리 변화
            sequential_distances = []
            for i in range(len(domain_ids) - 1):
                dist = combined_distances[i, i+1]
                sequential_distances.append(dist)
            
            ax4.plot(range(1, len(sequential_distances)+1), sequential_distances, 'o-', linewidth=2)
            ax4.set_xlabel('Domain Transition')
            ax4.set_ylabel('Distance')
            ax4.set_title('Sequential Domain Shift Magnitude')
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(range(1, len(sequential_distances)+1))
            ax4.set_xticklabels([f'D{domain_ids[i]}→D{domain_ids[i+1]}' 
                                for i in range(len(sequential_distances))])
            
            plt.suptitle(f'{scenario_name}: Domain Shift Analysis', fontsize=16)
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = self.output_dir / f'{scenario_name}_domain_shift_analysis.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Domain shift analysis 저장됨: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Domain shift analysis 생성 실패: {e}")
            return None
    
    def plot_continual_learning_summary(self,
                                       scenario_results: Dict[str, Dict],
                                       save_path: Optional[str] = None) -> str:
        """
        여러 시나리오의 Continual Learning 결과 종합 비교
        
        Args:
            scenario_results: 시나리오별 결과 딕셔너리
            save_path: 저장 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 입력 데이터 검증
            if not scenario_results:
                logger.warning("시나리오 결과가 비어있습니다")
                return None
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            scenario_names = list(scenario_results.keys())
            colors = COLORBLIND_PALETTE[:len(scenario_names)]
            
            # 1. 도메인별 정확도 비교
            for i, (scenario_name, results) in enumerate(scenario_results.items()):
                domain_names = results['domain_names']
                accuracies = results['final_accuracies']
                
                # 차원 일치 확인 및 보정
                min_len = min(len(domain_names), len(accuracies))
                domain_names = domain_names[:min_len]
                accuracies = accuracies[:min_len]
                
                if min_len > 0:  # 데이터가 있을 때만 플롯
                    ax1.plot(range(min_len), accuracies, 'o-', 
                            color=colors[i], label=scenario_name, linewidth=2, markersize=8)
            
            ax1.set_xlabel('Domain Index')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy Evolution Across Domains')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Forgetting Score 비교
            forgetting_data = []
            for scenario_name, results in scenario_results.items():
                # Forgetting score 계산 (간단화)
                accuracies = results.get('final_accuracies', [])
                if isinstance(accuracies, list) and len(accuracies) > 1:
                    forgetting = max(0, accuracies[0] - accuracies[-1])
                else:
                    forgetting = 0
                forgetting_data.append(forgetting)
            
            bars = ax2.bar(scenario_names, forgetting_data, color=colors[:len(scenario_names)])
            ax2.set_ylabel('Forgetting Score')
            ax2.set_title('Catastrophic Forgetting Comparison')
            ax2.grid(True, alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, forgetting_data):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 3. Retrieval 성능 비교
            retrieval_metrics = ['Top1', 'Top5']
            x = np.arange(len(scenario_names))
            width = 0.35
            
            top1_scores = [np.mean(results.get('final_top1_retrievals', [0])) 
                          for results in scenario_results.values()]
            top5_scores = [np.mean(results.get('final_top5_retrievals', [0])) 
                          for results in scenario_results.values()]
            
            ax3.bar(x - width/2, top1_scores, width, label='Top-1', color=colors[0])
            ax3.bar(x + width/2, top5_scores, width, label='Top-5', color=colors[1])
            
            ax3.set_xlabel('Scenario')
            ax3.set_ylabel('Retrieval Accuracy')
            ax3.set_title('Retrieval Performance Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(scenario_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 데이터 규모 vs 성능 산점도
            for i, (scenario_name, results) in enumerate(scenario_results.items()):
                total_samples = results.get('total_samples', 0)
                avg_accuracy = results.get('average_accuracy', 0)
                
                ax4.scatter(total_samples, avg_accuracy, s=200, 
                           color=colors[i], label=scenario_name, alpha=0.7)
                
                # 라벨 추가
                ax4.annotate(scenario_name, (total_samples, avg_accuracy),
                            xytext=(10, 10), textcoords='offset points')
            
            ax4.set_xlabel('Total Samples')
            ax4.set_ylabel('Average Accuracy')
            ax4.set_title('Data Scale vs Performance')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.suptitle('Continual Learning Performance Summary', fontsize=16)
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = self.output_dir / 'continual_learning_summary.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Continual learning summary 저장됨: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Continual learning summary 생성 실패: {e}")
            return None
    
    def create_paper_figures(self,
                            scenario_results: Dict[str, Dict],
                            domain_results: Dict[str, Dict[int, Dict]],
                            output_prefix: str = "paper_figure") -> List[str]:
        """
        논문용 통합 Figure 생성
        
        Args:
            scenario_results: 시나리오별 종합 결과
            domain_results: 시나리오별 도메인 결과
            output_prefix: 파일명 접두사
            
        Returns:
            List[str]: 생성된 파일 경로들
        """
        generated_files = []
        
        try:
            # Figure 1: 각 시나리오별 고급 t-SNE
            for scenario_name, domain_data in domain_results.items():
                tsne_path = self.plot_advanced_tsne(
                    domain_data, scenario_name,
                    save_path=self.output_dir / f'{output_prefix}_1_{scenario_name}_tsne.png'
                )
                if tsne_path:
                    generated_files.append(tsne_path)
            
            # Figure 2: Continual Learning 종합 비교
            summary_path = self.plot_continual_learning_summary(
                scenario_results,
                save_path=self.output_dir / f'{output_prefix}_2_continual_summary.png'
            )
            if summary_path:
                generated_files.append(summary_path)
            
            # Figure 3: 각 시나리오별 Domain Shift 분석
            for scenario_name, domain_data in domain_results.items():
                shift_path = self.plot_domain_shift_analysis(
                    domain_data, scenario_name,
                    save_path=self.output_dir / f'{output_prefix}_3_{scenario_name}_domain_shift.png'
                )
                if shift_path:
                    generated_files.append(shift_path)
            
            # Figure 4: 각 시나리오별 Confusion Matrix
            for scenario_name, domain_data in domain_results.items():
                cm_path = self.plot_confusion_matrices(
                    domain_data, scenario_name,
                    save_path=self.output_dir / f'{output_prefix}_4_{scenario_name}_confusion.png'
                )
                if cm_path:
                    generated_files.append(cm_path)
            
            logger.info(f"논문용 Figure {len(generated_files)}개 생성 완료!")
            return generated_files
            
        except Exception as e:
            logger.error(f"논문용 Figure 생성 실패: {e}")
            return generated_files


def create_visualizer(output_dir: str = 'visualizations') -> AdvancedVisualizer:
    """AdvancedVisualizer 인스턴스 생성"""
    return AdvancedVisualizer(output_dir)


if __name__ == "__main__":
    # 테스트 코드
    print("=== AdvancedVisualizer 테스트 ===")
    
    visualizer = create_visualizer('test_visualizations')
    print(f"시각화 모듈 초기화 완료: {visualizer.output_dir}")
    
    print("=== 테스트 완료 ===")
