from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from typing import List, Tuple, Union, Optional
import warnings


class TimeSeriesScaler:
    AVAILABLE_SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,  # 이상치에 강함
        'maxabs': MaxAbsScaler,  # 희소 데이터에 적합
    }
    
    def __init__(self, scaler_type: str = 'standard', **kwargs):
        """
        Args:
            scaler_type: 스케일러 타입 ('standard', 'minmax', 'robust', 'maxabs')
            **kwargs: 스케일러별 추가 파라미터
        """
        if scaler_type.lower() not in self.AVAILABLE_SCALERS:
            raise ValueError(f"Unknown scaler: {scaler_type}. "
                           f"Available: {list(self.AVAILABLE_SCALERS.keys())}")
        
        self.scaler_type = scaler_type.lower()
        self.scaler = self.AVAILABLE_SCALERS[self.scaler_type](**kwargs)
        self._fitted = False
    
    def fit(self, train_data: np.ndarray) -> 'TimeSeriesScaler':
        """훈련 데이터로 스케일러 학습"""
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)
        
        self.scaler.fit(train_data)
        self._fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """데이터 변환"""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        transformed = self.scaler.transform(data)
        
        if len(original_shape) == 1:
            transformed = transformed.flatten()
        
        return transformed
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """역변환 (정규화 해제)"""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        inverse = self.scaler.inverse_transform(data)
        
        if len(original_shape) == 1:
            inverse = inverse.flatten()
        
        return inverse
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """fit과 transform을 한번에 수행"""
        return self.fit(data).transform(data)


def apply_scaling(
    train_data: np.ndarray,
    val_data: np.ndarray, 
    test_data: np.ndarray,
    scaler_type: str = 'standard',
    return_scaler: bool = False,
    **scaler_kwargs
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], TimeSeriesScaler]]:
    """
    시계열 데이터에 스케일링 적용
    
    Args:
        train_data: 훈련 데이터 (스케일러 fitting용)
        val_data: 검증 데이터
        test_data: 테스트 데이터  
        scaler_type: 스케일러 타입
        return_scaler: 스케일러 객체 반환 여부
        **scaler_kwargs: 스케일러별 추가 파라미터
        
    Returns:
        [train_scaled, val_scaled, test_scaled] 또는 
        ([train_scaled, val_scaled, test_scaled], scaler)
    """
    # 입력 검증
    if not all(isinstance(data, np.ndarray) for data in [train_data, val_data, test_data]):
        warnings.warn("Input data should be numpy arrays")
    
    # 스케일러 초기화 및 학습
    scaler = TimeSeriesScaler(scaler_type, **scaler_kwargs)
    scaler.fit(train_data)
    
    # 데이터 변환
    scaled_data = [
        scaler.transform(train_data),
        scaler.transform(val_data), 
        scaler.transform(test_data)
    ]
    
    if return_scaler:
        return scaled_data, scaler
    return scaled_data