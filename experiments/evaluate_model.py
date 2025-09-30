"""
학습된 TextVibCLIP 모델 평가 스크립트
다양한 메트릭으로 모델 성능 평가
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.textvib_model_v2 import create_textvib_model_v2 as create_textvib_model
from src.data_loader import create_domain_dataloaders
from configs.model_config import DATA_CONFIG

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='TextVibCLIP 모델 평가')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='평가할 모델 체크포인트 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--subset', type=str, choices=['val', 'test'], default='test',
                       help='평가할 데이터셋')
    parser.add_argument('--save_results', action='store_true',
                       help='결과 저장 여부')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='결과 저장 디렉토리')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """모델 로딩"""
    logger.info(f"모델 로딩: {checkpoint_path}")
    
    model = create_textvib_model('continual')  # Evaluation용으로 continual mode
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("모델 로딩 완료")
    return model


def compute_embedding_similarity(text_embeddings: torch.Tensor, 
                               vib_embeddings: torch.Tensor) -> torch.Tensor:
    """임베딩 간 유사도 계산"""
    # L2 정규화
    text_embeddings = F.normalize(text_embeddings, dim=1)
    vib_embeddings = F.normalize(vib_embeddings, dim=1)
    
    # Cosine similarity
    similarity = torch.cosine_similarity(text_embeddings, vib_embeddings, dim=1)
    return similarity


def evaluate_domain(model, dataloader, device, domain_rpm):
    """단일 도메인 평가"""
    all_similarities = []
    all_metadata = []
    all_text_embeddings = []
    all_vib_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 배치를 디바이스로 이동
            if 'vibration' in batch:
                batch['vibration'] = batch['vibration'].to(device)
            
            # 임베딩 추출
            results = model(batch, return_embeddings=True)
            text_emb = results['text_embeddings']
            vib_emb = results['vib_embeddings']
            
            # 유사도 계산
            similarities = compute_embedding_similarity(text_emb, vib_emb)
            
            all_similarities.append(similarities.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_vib_embeddings.append(vib_emb.cpu())
            all_metadata.extend(batch['metadata'])
    
    # 결과 집계
    similarities = torch.cat(all_similarities, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    vib_embeddings = torch.cat(all_vib_embeddings, dim=0)
    
    # 간단한 분류 성능 (threshold 기반)
    predictions = (similarities > 0.5).long().numpy()
    targets = np.ones_like(predictions)  # 모든 쌍이 positive pair
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='binary'
    )
    
    return {
        'domain_rpm': domain_rpm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_similarity': similarities.mean().item(),
        'std_similarity': similarities.std().item(),
        'num_samples': len(similarities),
        'similarities': similarities,
        'text_embeddings': text_embeddings,
        'vib_embeddings': vib_embeddings,
        'metadata': all_metadata
    }


def evaluate_cross_domain_retrieval(domain_results):
    """Cross-domain retrieval 성능 평가"""
    logger.info("Cross-domain retrieval 평가 중...")
    
    # 모든 도메인의 임베딩 결합
    all_text_embeddings = []
    all_vib_embeddings = []
    domain_labels = []
    
    for domain_rpm, results in domain_results.items():
        text_emb = results['text_embeddings']
        vib_emb = results['vib_embeddings']
        
        all_text_embeddings.append(text_emb)
        all_vib_embeddings.append(vib_emb)
        domain_labels.extend([domain_rpm] * len(text_emb))
    
    # 전체 임베딩 행렬
    all_text = torch.cat(all_text_embeddings, dim=0)
    all_vib = torch.cat(all_vib_embeddings, dim=0)
    
    # Text → Vibration retrieval
    text_to_vib_similarity = torch.matmul(
        F.normalize(all_text, dim=1),
        F.normalize(all_vib, dim=1).t()
    )
    
    # Top-k accuracy 계산
    retrieval_results = {}
    for k in [1, 5, 10]:
        _, top_k_indices = torch.topk(text_to_vib_similarity, k=k, dim=1)
        
        # 정확한 매칭 확인 (diagonal elements)
        correct = 0
        for i in range(len(all_text)):
            if i in top_k_indices[i]:
                correct += 1
        
        retrieval_results[f'top_{k}_accuracy'] = correct / len(all_text)
    
    return retrieval_results


def plot_domain_performance(domain_results, save_path=None):
    """도메인별 성능 시각화"""
    domains = list(domain_results.keys())
    accuracies = [results['accuracy'] for results in domain_results.values()]
    similarities = [results['avg_similarity'] for results in domain_results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 정확도 플롯
    ax1.bar(range(len(domains)), accuracies)
    ax1.set_xlabel('Domain (RPM)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classification Accuracy by Domain')
    ax1.set_xticks(range(len(domains)))
    ax1.set_xticklabels([str(d) for d in domains])
    ax1.grid(True, alpha=0.3)
    
    # 평균 유사도 플롯
    ax2.bar(range(len(domains)), similarities)
    ax2.set_xlabel('Domain (RPM)')
    ax2.set_ylabel('Average Similarity')
    ax2.set_title('Average Embedding Similarity by Domain')
    ax2.set_xticks(range(len(domains)))
    ax2.set_xticklabels([str(d) for d in domains])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"성능 플롯 저장됨: {save_path}")
    else:
        plt.show()


def plot_embedding_space(domain_results, save_path=None):
    """임베딩 공간 시각화 (t-SNE)"""
    try:
        from sklearn.manifold import TSNE
        
        # 임베딩 수집
        all_text_embeddings = []
        all_vib_embeddings = []
        domain_labels = []
        
        for domain_rpm, results in domain_results.items():
            text_emb = results['text_embeddings']
            vib_emb = results['vib_embeddings']
            
            # 샘플링 (너무 많으면 시각화가 어려움)
            num_samples = min(100, len(text_emb))
            indices = torch.randperm(len(text_emb))[:num_samples]
            
            all_text_embeddings.append(text_emb[indices])
            all_vib_embeddings.append(vib_emb[indices])
            domain_labels.extend([f'Text-{domain_rpm}'] * num_samples)
            domain_labels.extend([f'Vib-{domain_rpm}'] * num_samples)
        
        # 전체 임베딩 결합
        all_text = torch.cat(all_text_embeddings, dim=0)
        all_vib = torch.cat(all_vib_embeddings, dim=0)
        all_embeddings = torch.cat([all_text, all_vib], dim=0).numpy()
        
        # t-SNE 실행
        logger.info("t-SNE 실행 중 (시간이 걸릴 수 있습니다)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        
        unique_labels = list(set(domain_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(domain_labels) if l == label]
            x = embeddings_2d[indices, 0]
            y = embeddings_2d[indices, 1]
            
            marker = 'o' if 'Text' in label else '^'
            plt.scatter(x, y, c=[colors[i]], label=label, marker=marker, alpha=0.7)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Embedding Space Visualization (t-SNE)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"임베딩 공간 시각화 저장됨: {save_path}")
        else:
            plt.show()
            
    except ImportError:
        logger.warning("scikit-learn이 설치되지 않아 t-SNE 시각화를 건너뜁니다.")


def main():
    """메인 평가 함수"""
    args = parse_args()
    
    logger.info("=== TextVibCLIP 모델 평가 ===")
    logger.info(f"모델: {args.model_checkpoint}")
    logger.info(f"데이터셋: {args.subset}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"디바이스: {device}")
    
    # 모델 로딩
    model = load_model(args.model_checkpoint, device)
    
    # 데이터로더 생성
    logger.info("데이터로더 생성 중...")
    domain_loaders = create_domain_dataloaders(batch_size=args.batch_size)
    
    # 도메인별 평가
    domain_results = {}
    overall_metrics = []
    
    logger.info("도메인별 평가 시작...")
    for domain_rpm, loaders in domain_loaders.items():
        dataloader = loaders[args.subset]
        
        logger.info(f"Domain {domain_rpm} 평가 중...")
        results = evaluate_domain(model, dataloader, device, domain_rpm)
        domain_results[domain_rpm] = results
        overall_metrics.append(results)
        
        logger.info(f"Domain {domain_rpm} 결과: "
                   f"Accuracy = {results['accuracy']:.4f}, "
                   f"F1 = {results['f1_score']:.4f}, "
                   f"Avg Similarity = {results['avg_similarity']:.4f}")
    
    # 전체 성능 계산
    avg_accuracy = np.mean([r['accuracy'] for r in overall_metrics])
    avg_f1 = np.mean([r['f1_score'] for r in overall_metrics])
    avg_similarity = np.mean([r['avg_similarity'] for r in overall_metrics])
    
    logger.info("=== 전체 성능 ===")
    logger.info(f"평균 정확도: {avg_accuracy:.4f}")
    logger.info(f"평균 F1 Score: {avg_f1:.4f}")
    logger.info(f"평균 유사도: {avg_similarity:.4f}")
    
    # Cross-domain retrieval 평가
    retrieval_results = evaluate_cross_domain_retrieval(domain_results)
    logger.info("=== Cross-domain Retrieval ===")
    for metric, value in retrieval_results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 결과 저장
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 성능 플롯
        performance_plot_path = os.path.join(args.output_dir, 'domain_performance.png')
        plot_domain_performance(domain_results, performance_plot_path)
        
        # 임베딩 공간 시각화
        embedding_plot_path = os.path.join(args.output_dir, 'embedding_space.png')
        plot_embedding_space(domain_results, embedding_plot_path)
        
        # 결과 저장
        evaluation_results = {
            'domain_results': domain_results,
            'overall_metrics': {
                'avg_accuracy': avg_accuracy,
                'avg_f1': avg_f1,
                'avg_similarity': avg_similarity
            },
            'retrieval_results': retrieval_results,
            'model_checkpoint': args.model_checkpoint,
            'evaluation_subset': args.subset
        }
        
        results_path = os.path.join(args.output_dir, 'evaluation_results.pth')
        torch.save(evaluation_results, results_path)
        logger.info(f"평가 결과 저장됨: {results_path}")
    
    logger.info("=== 모델 평가 완료 ===")


if __name__ == "__main__":
    main()
