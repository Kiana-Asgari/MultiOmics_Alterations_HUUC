import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import re

class ConfigManager:
    def __init__(self):
        self.config_path = Path(__file__).parent / "model_configs.yaml"
        self.marker_config_path = Path(__file__).parent / "marker_configs.yaml"
        self._config = self._load_config()
        self._marker_config = self._load_marker_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _load_marker_config(self) -> Dict[str, Any]:
        if not self.marker_config_path.exists():
            raise FileNotFoundError(f"Marker config file not found: {self.marker_config_path}")
        
        with open(self.marker_config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @property
    def countries(self):
        return tuple(self._config['data']['countries'])
    
    @property
    def data_types(self):
        return tuple(self._config['data']['data_types'])
    
    @property
    def features_types(self) -> list:
        return self._config['data']['columns']
    @property
    def genom_types(self):
        return tuple(self._config['data']['genom_types'])
    
    @property
    def diseases(self):
        return tuple(self._config['diseases'])
    
    @property
    def neonatal_sources(self):
        sources = self._config['neonatal_sources']
        return (sources['year_collected'], sources['genom_type'], sources['patient_type'], sources['target_column'])
 
    



    
    def get_file_mapping(self, country: str, data_type: str=None, genom_type: str=None) -> str:
        """Get the file name for a given country and data type."""
        if (country == "Bangladesh" or country == "Zimbabwe") and (data_type == "Questionnaire" or data_type == "Metabolome"):
            file_name = self._config['data']['file_mapping'][country][data_type]
        elif country == "Zambia" and (data_type == "2019-2020" or data_type == "2019-2021"):
            file_name = self._config['data']['file_mapping'][country][data_type]
        else:
            file_name = self._config['data']['file_mapping'][country][genom_type][data_type]

        return file_name


    def extract_features(self, country: str, data_type: str, feature_type: str, patient_type: str) -> list:
        """Get columns for a given country, data type, and column type."""
        return self._config['data'][feature_type][country][data_type][patient_type]

    def get_column_standardization(self) -> dict:
        """Get column standardization mapping."""
        return self._config['data']['column_standardization']

    def growth_chart(self) -> Dict[str, list]:
        """Get growth chart data."""
        return self._config['growth_chart']
    
    def get_neonatal_marker_full_name(self, marker_code: str) -> str:
        """
        Get the full name of a neonatal marker given its code.
        Returns the original code if no mapping is found.
        
        Args:
            marker_code: The marker code (e.g., 'ala', 'arg', 'c02', 'C10.1')
        
        Returns:
            The full name of the marker if found, otherwise the original marker code
        """
        original_code = marker_code
        if marker_code is None:
            return original_code

        code = str(marker_code).strip().lower()

        # 1) Prefer short display names for neonatal biomarkers (publication-friendly).
        neonate_map = self._marker_config.get("NEONATES_BIOMARKER_NAME_MAP", {}) or {}

        # normalize common variants:
        # - allow ":" and "." variants (e.g., C8:1 vs c8.1)
        # - allow compact acylcarnitine codes (e.g., c81 -> c8.1, c101 -> c10.1)
        candidates = [code, code.replace(":", ".")]
        if re.fullmatch(r"c\d{2,3}", code):
            candidates.append(f"c{code[1:-1]}.{code[-1]}")

        for cand in candidates:
            if cand in neonate_map:
                return neonate_map[cand]

        # 2) Fall back to verbose Kenya mapping (if present).
        kenya_map = self._marker_config.get("Kenya_neonatal_markers_mapping", {}) or {}
        kenya_key = code.replace(".", "")
        if kenya_key in kenya_map:
            return kenya_map[kenya_key]

        return original_code

    def get_maternal_feature_full_name(self, feature_code: str) -> str:
        """
        Get the display name of a maternal feature given its code.
        Returns the original code if no mapping is found.
        """
        original = feature_code
        if feature_code is None:
            return original
        code = str(feature_code).strip().lower()
        maternal_map = self._marker_config.get("MATERNAL_FEATURE_NAME_MAP", {}) or {}

        # Accept common variants across lipidome/metabolome pipelines.
        # e.g. geranyl_pp vs geranyl.pp
        candidates = [code]
        if "_" in code:
            candidates.append(code.replace("_", "."))
        if "." in code:
            candidates.append(code.replace(".", "_"))

        for cand in candidates:
            if cand in maternal_map:
                return maternal_map[cand]
        return original
    
    def get_all_markers(self, country: str = "Kenya", patient_type: str = "neonatal") -> Dict[str, str]:
        """
        Get all marker mappings for a given country and patient type.
        
        Args:
            country: The country name (default: 'Kenya')
            patient_type: The patient type (default: 'neonatal')
        
        Returns:
            Dictionary mapping marker codes to full names
        """
        mapping_key = f"{country}_{patient_type}_markers_mapping"
        
        if mapping_key not in self._marker_config:
            raise KeyError(f"Marker mapping '{mapping_key}' not found in marker_configs.yaml")
        
        return self._marker_config[mapping_key]

    

# Global instance
config = ConfigManager()