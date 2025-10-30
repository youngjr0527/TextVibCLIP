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
from matplotlib.colors import LinearSegmentedColormap

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

        # UOS 7-클래스 색상/마커 (조합 라벨: H_H,H_B,H_IR,H_OR,L_H,U_H,M_H)
        self.uos_colors = {
            'H_H': '#1f77b4',  # blue
            'H_B': '#ff7f0e',  # orange
            'H_IR': '#2ca02c', # green
            'H_OR': '#d62728', # red
            'L_H': '#9467bd',  # purple
            'U_H': '#8c564b',  # brown
            'M_H': '#e377c2'   # pink
        }
        self.uos_markers = {
            'H_H': 'o',
            'H_B': 's',
            'H_IR': '^',
            'H_OR': 'D',
            'L_H': 'v',
            'U_H': 'P',
            'M_H': 'X'
        }
        
        # 베어링 타입별 마커
        self.bearing_markers = {
            'N204': 'o',      # 원형
            'NJ204': 's',     # 사각형
            '6204': '^',      # 삼각형
            '30204': 'D',     # 다이아몬드
            'deep_groove_ball': 'o',  
            'SKF': 'o'        
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
        
        # 스타일 선택 헬퍼 (UOS 7클래스)
        def _style_for(cond: str):
            if cond in self.uos_colors:
                return self.uos_colors[cond], self.uos_markers[cond]
            return self.bearing_colors.get(cond, self.colors['neutral']), self.condition_markers.get(cond, 'o')

        # 1. 베어링 상태별 분포 (Text만) - 좌상단
        ax2 = axes[0, 0]
        text_mask = np.array(modality_labels) == 'Text'
        text_coords = embeddings_2d[text_mask]
        text_conditions = np.array(condition_labels)[text_mask]
        
        for condition in np.unique(text_conditions):
            mask = text_conditions == condition
            color, marker = _style_for(condition)
            ax2.scatter(text_coords[mask, 0], text_coords[mask, 1],
                        c=color,
                        alpha=0.85, s=70, label=f'{condition}',
                        marker=marker,
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
            color, marker = _style_for(condition)
            ax3.scatter(vib_coords[mask, 0], vib_coords[mask, 1],
                        c=color,
                        alpha=0.85, s=70, label=f'{condition}',
                        marker=marker,
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

    def create_continual_learning_curve(self,
                                        domain_names: List[str],
                                        accuracies: List[float],
                                        scenario_name: str,
                                        save_name: str = "continual_learning_curve") -> str:
        """Continual Learning 성능 곡선 시각화
        
        Args:
            domain_names: 도메인 이름 리스트 (예: ['600RPM', '800RPM', ...])
            accuracies: 각 도메인의 최종 정확도 리스트
            scenario_name: 시나리오 이름
            save_name: 저장 파일명
            
        Returns:
            저장 경로
        """
        logger.info(f"Continual learning curve 생성 중: {scenario_name}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(domain_names))
        
        # 메인 라인 플롯
        ax.plot(x, accuracies, marker='o', linewidth=2.5, markersize=10,
                color=self.colors['primary'], label='Retrieval Accuracy')
        
        # 각 포인트에 정확도 표시
        for i, (domain, acc) in enumerate(zip(domain_names, accuracies)):
            ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        # 평균선 추가
        avg_acc = np.mean(accuracies)
        ax.axhline(avg_acc, color=self.colors['secondary'], linestyle='--',
                  linewidth=2, label=f'Average: {avg_acc:.1%}', alpha=0.7)
        
        # 스타일링
        ax.set_xlabel('Domain Sequence', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'Continual Learning Performance - {scenario_name}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(domain_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        
        # 배경색
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{scenario_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Continual learning curve 저장 완료: {save_path}")
        return save_path

    def create_forgetting_heatmap(self,
                                  domain_names: List[str],
                                  accuracy_matrix: np.ndarray,
                                  scenario_name: str,
                                  save_name: str = "forgetting_heatmap") -> str:
        """Forgetting Analysis Heatmap 시각화 (각 행 평균 포함)
        
        Args:
            domain_names: 도메인 이름 리스트
            accuracy_matrix: (n_domains, n_domains) 정확도 행렬
                            [i, j] = domain i 학습 후 domain j에서의 정확도
            scenario_name: 시나리오 이름
            save_name: 저장 파일명
            
        Returns:
            저장 경로
        """
        logger.info(f"Continual learning heatmap 생성 중: {scenario_name}")
        
        # 🎯 각 행의 평균 계산 (Average Accuracy per Stage)
        row_averages = []
        for i in range(len(domain_names)):
            row_values = accuracy_matrix[i, :]
            valid_values = row_values[~np.isnan(row_values)]
            if len(valid_values) > 0:
                row_avg = np.mean(valid_values)
                row_averages.append(row_avg)
            else:
                row_averages.append(np.nan)
        
        # 확장된 행렬 생성 (원본 + 평균 열)
        n_domains = len(domain_names)
        extended_matrix = np.full((n_domains, n_domains + 1), np.nan)
        extended_matrix[:, :n_domains] = accuracy_matrix
        extended_matrix[:, n_domains] = row_averages  # 마지막 열에 평균 추가
        
        # Figure 생성 (평균 열을 위해 더 넓게, 간격 조정)
        fig = plt.figure(figsize=(14, 8))
        
        # GridSpec으로 메인 heatmap과 평균 열 분리
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 2, width_ratios=[n_domains, 1], wspace=0.15)
        
        ax_main = fig.add_subplot(gs[0])  # 메인 heatmap
        ax_avg = fig.add_subplot(gs[1])  # 평균 열 (독립적 Y축)
        
        cmap_custom = LinearSegmentedColormap.from_list(
            'white_to_lightgreen',
            ['#ffffff', '#f7c6ba', '#fa7e5f']   # 필요하면 마지막 색 더 밝게(예: #d6f5c2)
        )

        # 🎯 메인 Heatmap 생성
        accuracy_matrix_percent = accuracy_matrix * 100
        im_main = ax_main.imshow(accuracy_matrix_percent, cmap=cmap_custom, aspect='auto',
                                vmin=0, vmax=100, interpolation='nearest')
        
        # 메인 heatmap 축 설정
        ax_main.set_xticks(np.arange(n_domains))
        ax_main.set_yticks(np.arange(n_domains))
        ax_main.set_xticklabels(domain_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax_main.set_yticklabels(domain_names, rotation=0, ha='right', fontsize=11, fontweight='bold')
        
        ax_main.set_xlabel('Test Domain', fontsize=13, fontweight='bold', labelpad=10)
        ax_main.set_ylabel('Domain Shift Progress', fontsize=13, fontweight='bold', labelpad=15)
        
        # 메인 heatmap 셀 값 표시 (모두 검은색 볼드)
        for i in range(n_domains):
            for j in range(n_domains):
                if not np.isnan(accuracy_matrix[i, j]):
                    percent_val = accuracy_matrix[i, j] * 100
                    ax_main.text(j, i, f'{percent_val:.2f}%',
                               ha='center', va='center', color='black',
                               fontsize=13, fontweight='bold')
        
        # 🎯 Stage Average 열 생성
        row_avg_matrix = np.array(row_averages).reshape(-1, 1)
        row_avg_percent = row_avg_matrix * 100
        im_avg = ax_avg.imshow(row_avg_percent, cmap=cmap_custom, aspect='auto',
                              vmin=0, vmax=100, interpolation='nearest')
        
        # 평균 열 축 설정
        ax_avg.set_xticks([])  # X축 틱 제거
        ax_avg.set_yticks([])  # Y축 틱 제거 (오른쪽 열만)
        ax_avg.set_xlabel('')
        
        # 평균 열 제목 추가 
        ax_avg.text(0.5, 1.08, 'Average Accuracy', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   transform=ax_avg.transAxes)
        
        # 평균 열 값 표시 (모두 검은색 볼드)
        for i in range(n_domains):
            if not np.isnan(row_averages[i]):
                percent_val = row_averages[i] * 100
                ax_avg.text(0, i, f'{percent_val:.1f}%',
                          ha='center', va='center', color='black',
                          fontsize=15, fontweight='bold')
        
        # 전체 타이틀
        fig.suptitle(f'Continual learning Analysis - {scenario_name}',
                    fontsize=15, fontweight='bold', y=0.98)
        
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{scenario_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Forgetting heatmap 저장 완료: {save_path}")
        logger.info(f"각 행 평균: {[f'{avg*100:.1f}%' for avg in row_averages if not np.isnan(avg)]}")
        return save_path
    
    def create_replay_comparison_plot(self,
                                    replay_results: Dict,
                                    replay_free_results: Dict,
                                    scenario_name: str,
                                    save_name: str = "replay_comparison") -> str:
        """
        Replay vs Replay-Free 성능 비교 선 그래프 생성
        
        Args:
            replay_results: Replay buffer 사용 결과
            replay_free_results: Replay-free 결과
            scenario_name: 시나리오 이름
            save_name: 저장 파일명
        """
        if not replay_results or not replay_free_results:
            logger.warning("Replay 비교 결과가 부족합니다.")
            return ""
        
        # 데이터 추출
        replay_accuracies = replay_results.get('stage_accuracies', [])
        replay_free_accuracies = replay_free_results.get('stage_accuracies', [])
        
        # 도메인 이름 (첫 번째 결과에서 추출)
        domain_names = replay_results.get('domain_names', [])
        
        # 유효한 데이터만 필터링 (None 값 제거)
        valid_indices = []
        valid_replay = []
        valid_replay_free = []
        valid_domains = []
        
        for i, (r_acc, rf_acc) in enumerate(zip(replay_accuracies, replay_free_accuracies)):
            if r_acc is not None and rf_acc is not None:
                valid_indices.append(i)
                valid_replay.append(r_acc)
                valid_replay_free.append(rf_acc)  # 수정: replay_free_accuracies 사용
                valid_domains.append(domain_names[i] if i < len(domain_names) else f"Domain {i+1}")
        
        if not valid_replay:
            logger.warning("유효한 비교 데이터가 없습니다.")
            return ""
        
        # Figure 생성 (단일 subplot, 적절한 aspect ratio)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Replay Buffer Ablation Study - {scenario_name}', 
                    fontsize=16, fontweight='bold')
        
        # Accuracy 비교
        x_pos = range(len(valid_replay))
        
        # 선 그래프 그리기
        line1 = ax.plot(x_pos, valid_replay, 
                       marker='o', linewidth=2.5, markersize=8,
                       color=self.colors['primary'], label='With Replay Buffer',
                       markerfacecolor='white', markeredgewidth=2)
        
        line2 = ax.plot(x_pos, valid_replay_free,
                       marker='s', linewidth=2.5, markersize=8,
                       color=self.colors['secondary'], label='Replay-Free',
                       markerfacecolor='white', markeredgewidth=2)
        
        ax.set_title('Stage-wise Accuracy Comparison', fontweight='bold', fontsize=14)
        ax.set_xlabel('Training Stage', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_domains, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Y축 범위 설정 (0-100%)
        ax.set_ylim(0, 100)
        
        # 값 표시
        for i, (r_acc, rf_acc) in enumerate(zip(valid_replay, valid_replay_free)):
            ax.annotate(f'{r_acc:.1f}%', (i, r_acc), 
                       textcoords="offset points", xytext=(0,10), ha='center',
                       fontsize=10, color=self.colors['primary'], fontweight='bold')
            ax.annotate(f'{rf_acc:.1f}%', (i, rf_acc),
                       textcoords="offset points", xytext=(0,-15), ha='center',
                       fontsize=10, color=self.colors['secondary'], fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}_{scenario_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Replay comparison plot 저장 완료: {save_path}")
        return save_path

def create_visualizer(output_dir: str) -> PaperVisualizer:
    """PaperVisualizer 인스턴스 생성 (기존 호환성 유지)"""
    return PaperVisualizer(output_dir)
