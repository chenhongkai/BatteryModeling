#%%
from .DFNJTFP2D import DFNJTFP2D
from .LPJTFP2D import LPJTFP2D
from .ELPJTFP2D import ELPJTFP2D
from .tools import (LumpedParameters, EnhancedLumpedParameters, ConservativeLumpedParameters,
                    set_matplotlib, DiagonalSliceRavel)

try:
    import mph
    from .COMSOLP2D import COMSOLP2D
except ImportError:
    # print("mph库未安装")
    pass
