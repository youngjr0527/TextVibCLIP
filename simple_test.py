#!/usr/bin/env python3
"""
가장 간단한 테스트: 첫 번째 도메인만 단순 분류
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.textvib_model import create_textvib_model
from src.data_loader import create_first_domain_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_classification_test():
    """간단한 분류 테스트"""
    logger.info("=== 간단한 분류 테스트 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 생성 (auxiliary head만 사용)
    model = create_textvib_model('first_domain')
    model.to(device)
    
    # 간단한 분류기 추가
    simple_classifier = nn.Linear(256, 7).to(device)  # 진동 임베딩 → 7클래스
    
    # 데이터로더
    dataloader = create_first_domain_dataloader(
        data_dir='data_scenario1',
        dataset_type='uos',
        batch_size=32,
        num_workers=0
    )
    
    # 옵티마이저
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': simple_classifier.parameters()}
    ], lr=1e-3)
    
    # 학습
    model.train()
    simple_classifier.train()
    
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:  # 50배치만
                break
                
            # 배치 준비
            vibration = batch['vibration'].to(device)
            labels = batch['labels'].to(device)
            
            if labels.dim() == 2:
                labels = labels[:, 0]  # 주 분류만 사용
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                vib_embeddings = model.encode_vibration(vibration)
            
            # 간단한 분류
            logits = simple_classifier(vib_embeddings)
            loss = F.cross_entropy(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # 통계
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                acc = correct / total if total > 0 else 0
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.4f}")
        
        epoch_acc = correct / total if total > 0 else 0
        logger.info(f"Epoch {epoch+1} 완료: Loss={total_loss/(batch_idx+1):.4f}, Acc={epoch_acc:.4f}")
        
        # 50% 이상이면 성공
        if epoch_acc > 0.5:
            logger.info(f"✅ 성공! 간단한 분류기로 {epoch_acc:.1%} 달성")
            return True
    
    logger.error(f"❌ 실패: 최종 정확도 {epoch_acc:.1%}")
    return False

if __name__ == "__main__":
    success = simple_classification_test()
    if success:
        print("\n✅ 데이터와 모델은 정상 - TextVibCLIP 구현에 문제가 있음")
    else:
        print("\n❌ 데이터 자체에 문제가 있거나 모델 아키텍처 문제")
