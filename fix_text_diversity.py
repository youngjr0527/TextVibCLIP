#!/usr/bin/env python3
"""
텍스트 다양성 개선 - 클래스별 고유 키워드 강화
"""

def generate_improved_text_description(metadata):
    """개선된 텍스트 설명 생성 - 클래스별 차별화"""
    
    rotating_comp = metadata['rotating_component']
    bearing_cond = metadata['bearing_condition']
    
    # 🎯 CRITICAL FIX: 클래스별 고유 키워드 강화
    if rotating_comp == 'H' and bearing_cond == 'H':
        # 완전 정상
        templates = [
            "Perfect bearing condition with no defects",
            "Healthy machinery in optimal operating state", 
            "Normal bearing operation without faults",
            "Flawless bearing performance",
            "Pristine bearing condition"
        ]
    elif rotating_comp == 'H' and bearing_cond == 'B':
        # 볼 결함
        templates = [
            "Ball element damage detected",
            "Rolling element deterioration present",
            "Spherical component failure identified",
            "Ball bearing surface defect",
            "Rolling ball wear pattern"
        ]
    elif rotating_comp == 'H' and bearing_cond == 'IR':
        # 내륜 결함
        templates = [
            "Inner race crack formation",
            "Internal ring deterioration",
            "Inner raceway surface damage",
            "Inner ring structural failure",
            "Internal bearing track defect"
        ]
    elif rotating_comp == 'H' and bearing_cond == 'OR':
        # 외륜 결함
        templates = [
            "Outer race structural damage",
            "External ring deterioration",
            "Outer raceway surface defect",
            "Outer ring crack development",
            "External bearing track failure"
        ]
    elif rotating_comp == 'L' and bearing_cond == 'H':
        # 회전체 느슨함
        templates = [
            "Mechanical looseness in shaft assembly",
            "Loose coupling connection detected",
            "Shaft mounting instability",
            "Mechanical joint looseness",
            "Assembly connection weakness"
        ]
    elif rotating_comp == 'U' and bearing_cond == 'H':
        # 회전체 불균형
        templates = [
            "Rotor mass imbalance condition",
            "Unbalanced rotating assembly",
            "Dynamic imbalance detected",
            "Rotor weight distribution error",
            "Centrifugal force imbalance"
        ]
    elif rotating_comp == 'M' and bearing_cond == 'H':
        # 회전체 정렬불량
        templates = [
            "Shaft misalignment detected",
            "Angular alignment error",
            "Parallel misalignment condition",
            "Shaft positioning defect",
            "Mechanical alignment failure"
        ]
    else:
        templates = ["Unknown bearing condition"]
    
    import random
    return random.choice(templates)

# 테스트
if __name__ == "__main__":
    test_metadata = [
        {'rotating_component': 'H', 'bearing_condition': 'H'},
        {'rotating_component': 'H', 'bearing_condition': 'B'},
        {'rotating_component': 'H', 'bearing_condition': 'IR'},
        {'rotating_component': 'L', 'bearing_condition': 'H'},
    ]
    
    print("개선된 텍스트 생성 예시:")
    for i, meta in enumerate(test_metadata):
        text = generate_improved_text_description(meta)
        print(f"{i+1}. {text}")
