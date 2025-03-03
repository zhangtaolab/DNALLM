import pytest
from dnallm.finetune.models.base import BaseDNAModel

def test_base_model_abstract_methods():
    """Test that BaseDNAModel cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseDNAModel()

def test_model_interface():
    """Test model interface requirements"""
    class TestModel(BaseDNAModel):
        def get_model(self):
            return None
        
        def preprocess(self, sequences):
            return {}
    
    model = TestModel()
    assert hasattr(model, 'get_model')
    assert hasattr(model, 'preprocess') 