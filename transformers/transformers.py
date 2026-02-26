import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Base(BaseEstimator, TransformerMixin):
    pass

class CompanyBinning(Base):
    def __init__(self):
        self.feature_name_ = None
        self.class_1 = ['Lenovo', 'Asus', 'Dell', 'HP', 'Acer', 'Toshiba', 'Xiaomi', 'Fujitsu']
        self.class_2 = ['Chuwi', 'Mediacom', 'Vero']
        self.class_3 = ['Apple', 'Samsung', 'MSI', 'LG', 'Microsoft', 'Google']
        self.class_4 = ['Razer']

    def fit(self, X, y=None):
        self.feature_name_ = 'bin_Company'
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X['Company'] = X['Company'].fillna('')
        conditions = [
            X['Company'].isin(self.class_1),
            X['Company'].isin(self.class_2),
            X['Company'].isin(self.class_3),
            X['Company'].isin(self.class_4)
        ]
        choices = ['class_1', 'class_2', 'class_3', 'class_4']
        X[self.feature_name_] = pd.Categorical(np.select(conditions, choices, default='Rare')).codes
        return X[[self.feature_name_]]

class TypeNameEncoder(Base):
    def __init__(self):
        self.feature_name_ = None

    def fit(self, X, y=None):
        self.feature_name_ = ['is_Netbook', 'is_Notebook', 'Gaming_or_Workstation', 'Ultrabook_or_2in1']
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.Series):
            X = X.to_frame()
        mapping = {
            'is_Netbook': ['Netbook'],
            'is_Notebook': ['Notebook'],
            'Gaming_or_Workstation': ['Gaming', 'Workstation'],
            'Ultrabook_or_2in1': ['Ultrabook', '2 in 1 Convertible']
        }
        for col, values in mapping.items():
            X[col] = X['TypeName'].isin(values).astype(int)
        return X[self.feature_name_]

class InchesBinning(Base):
    def __init__(self):
        self.feature_name_ = None

    def fit(self, X, y=None):
        self.feature_name_ = 'bin_Inches'
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            series = pd.to_numeric(X, errors='coerce')
        else:
            series = pd.to_numeric(X.iloc[:,0], errors='coerce')
        bins = [0, 12, 16, 18, np.inf]
        labels = ['Small', 'Normal', 'Large', 'Rare']
        codes = pd.cut(series, bins=bins, labels=labels, include_lowest=True).cat.codes
        return pd.DataFrame({self.feature_name_: codes})

class ScreenResolutionEncoder(Base):
    def __init__(self):
        self.feature_names_ = None
    def fit(self, X, y=None):
        self.feature_names_ = ['Resolution_Width','Resolution_Height','Pixel_Count','is_IPS','is_Touchscreen']
        return self
    def _extract_resolution(self, text):
        match = re.search(r'(\d+)x(\d+)', text)
        return (int(match.group(1)), int(match.group(2))) if match else (0,0)
    def transform(self, X):
        X = X.copy()
        widths, heights = zip(*X.iloc[:,0].apply(self._extract_resolution))
        X['Resolution_Width'] = widths
        X['Resolution_Height'] = heights
        X['Pixel_Count'] = X['Resolution_Width'] * X['Resolution_Height']
        X['is_IPS'] = X.iloc[:,0].str.contains('IPS', case=False).astype(int)
        X['is_Touchscreen'] = X.iloc[:,0].str.contains('Touchscreen', case=False).astype(int)
        return X[self.feature_names_]

class CpuEncoder(Base):
    def __init__(self):
        self.feature_names_ = None
    def fit(self, X, y=None):
        self.feature_names_ = ['is_Intel','is_AMD','Tier_Level','CPU_GHz','is_HQ','is_U','is_HK']
        return self
    def _extract_ghz(self, text):
        match = re.search(r'(\d+\.?\d*)GHz', text)
        return float(match.group(1)) if match else 0.0
    def _extract_tier(self, text):
        text = text.upper()
        if 'I7' in text or 'RYZEN 7' in text: return 4
        elif 'I5' in text or 'RYZEN 5' in text: return 3
        elif 'I3' in text or 'RYZEN 3' in text: return 2
        elif any(x in text for x in ['CELERON','PENTIUM','A6','A8','A9','A10','A12']): return 1
        else: return 0
    def transform(self, X):
        X = X.copy()
        s = X.iloc[:,0].fillna('')
        X['is_Intel'] = s.str.contains('Intel', case=False).astype(int)
        X['is_AMD'] = s.str.contains('AMD', case=False).astype(int)
        X['CPU_GHz'] = s.apply(self._extract_ghz)
        X['Tier_Level'] = s.apply(self._extract_tier)
        X['is_HQ'] = s.str.contains('HQ').astype(int)
        X['is_U'] = s.str.contains(' U').astype(int)
        X['is_HK'] = s.str.contains('HK').astype(int)
        return X[self.feature_names_]

class RamBinning(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_name_ = None
    def fit(self, X, y=None):
        if 'Ram' not in X.columns:
            raise ValueError("Ram column not found")
        self.feature_name_ = 'bin_Ram'
        return self
    def transform(self, X):
        if self.feature_name_ is None:
            raise ValueError("Transformer not fitted yet")
        X = X.copy()
        X['Ram'] = X['Ram'].str.extract(r'(\d+)\s*GB', flags=re.IGNORECASE).astype(float)
        conditions = [
            X[['Ram']] <= 6,
            (X[['Ram']] > 6) & (X[['Ram']] <= 16),
            X[['Ram']] == 32
        ]
        choices = ['S', 'M', 'L']
        X[[self.feature_name_]] = np.select(
            conditions,
            choices,
            default='Rare'
        )
        return X[[self.feature_name_]]

        
class MemoryEncoder(Base):
    def __init__(self):
        self.feature_names_ = [
            'SSD_GB','HDD_GB','Flash_GB','Hybrid_GB',
            'Total_Storage_GB','Has_SSD','Has_HDD'
        ]
    def fit(self, X, y=None): return self
    def _convert_to_gb(self, size, unit):
        try: size=float(size)
        except: return 0
        return int(size*1024) if unit.upper()=='TB' else int(size)
    def _parse_memory(self, memory_str):
        ssd=hdd=flash=hybrid=0
        if not isinstance(memory_str,str): return ssd,hdd,flash,hybrid
        for part in memory_str.split('+'):
            part=part.strip()
            match=re.search(r'(\d+\.?\d*)\s*(GB|TB)', part, re.IGNORECASE)
            if not match: continue
            size,unit=match.groups()
            size_gb=self._convert_to_gb(size,unit)
            if 'SSD' in part.upper(): ssd+=size_gb
            elif 'HDD' in part.upper(): hdd+=size_gb
            elif 'FLASH' in part.upper(): flash+=size_gb
            elif 'HYBRID' in part.upper(): hybrid+=size_gb
        return ssd,hdd,flash,hybrid
    def transform(self, X):
        if isinstance(X,pd.Series): series=X
        else: series=X.iloc[:,0]
        ssd_list,hdd_list,flash_list,hybrid_list=[],[],[],[]
        for mem in series: ssd,hdd,flash,hybrid=self._parse_memory(mem); ssd_list.append(ssd); hdd_list.append(hdd); flash_list.append(flash); hybrid_list.append(hybrid)
        df=pd.DataFrame({'SSD_GB':ssd_list,'HDD_GB':hdd_list,'Flash_GB':flash_list,'Hybrid_GB':hybrid_list})
        df['Total_Storage_GB']=df[['SSD_GB','HDD_GB','Flash_GB','Hybrid_GB']].sum(axis=1)
        df['Has_SSD']=(df['SSD_GB']>0).astype(int)
        df['Has_HDD']=(df['HDD_GB']>0).astype(int)
        return df[self.feature_names_]

class GpuEncoder(Base):
    def __init__(self):
        self.feature_names_ = ['is_Intel_GPU','is_Nvidia_GPU','is_AMD_GPU','is_ARM_GPU','is_Dedicated','GPU_Tier']
    def fit(self,X,y=None): return self
    def _extract_tier(self,text):
        t=text.replace(' ','').upper()
        if any(x in t for x in ['GTX1080','GTX1070','GTX1060','QUADRO']): return 2
        elif any(x in t for x in ['GTX1050','GTX1050TI','MX','RX','R5','R7']): return 1
        else: return 0
    def transform(self,X):
        if isinstance(X,pd.Series): s=X.fillna('')
        else: s=X.iloc[:,0].fillna('')
        df=pd.DataFrame()
        df['is_Intel_GPU']=s.str.contains('Intel',case=False).astype(int)
        df['is_Nvidia_GPU']=s.str.contains('Nvidia',case=False).astype(int)
        df['is_AMD_GPU']=s.str.contains('AMD',case=False).astype(int)
        df['is_ARM_GPU']=s.str.contains('ARM',case=False).astype(int)
        df['is_Dedicated'] = ((df['is_Nvidia_GPU']|df['is_AMD_GPU']) & s.str.contains('GTX|Quadro|Pro|RX|MX',case=False)).astype(int)
        df['GPU_Tier']=s.apply(self._extract_tier)
        return df[self.feature_names_]

class OpSysEncoder(Base):
    def __init__(self):
        self.feature_names_=['is_mac','is_Android','is_Linux', 'is_Windows']
    def fit(self,X,y=None): return self
    def transform(self,X):
        if isinstance(X,pd.Series): s=X.fillna('')
        else: s=X.iloc[:,0].fillna('')
        df=pd.DataFrame()
        df['is_mac'] = s.isin(['macOS','Mac OS X']).astype(int)
        df['is_Android'] = s.isin(['Android']).astype(int)
        df['is_Linux'] = s.isin(['Linux']).astype(int)
        df['is_Windows'] = s.isin(['Windows 10']).astype(int)
        return df[self.feature_names_]

class WeightExtract(Base):
    def __init__(self):
        self.feature_name_='extract_weight'
    def fit(self,X,y=None): return self
    def transform(self,X):
        if isinstance(X,pd.Series): series=X
        else: series=X.iloc[:,0]
        def _parse_weight(val):
            if pd.isna(val): return np.nan
            val=str(val).strip()
            m_kg=re.match(r'(\d+(?:\.\d+)?)\s*kg',val,re.IGNORECASE)
            if m_kg: return float(m_kg.group(1))
            m_g=re.match(r'(\d+(?:\.\d+)?)\s*g',val,re.IGNORECASE)
            if m_g: return float(m_g.group(1))/1000
            m_n=re.match(r'(\d+(?:\.\d+)?)',val)
            if m_n: return float(m_n.group(1))
            return np.nan
        return pd.DataFrame({self.feature_name_: series.apply(_parse_weight)})