## Details for the environment

'''
DIFUSCO Env. Setting


PyTorch
1. Pytorch hompage에서 cuda에 맞는 거 install
2. If the following error exist: “operator torchvision::nms does not exist”
3. Then, it is the problem of version compatibility b/w PyTorch and torch vision
4. Going to https://pytorch.org/get-started/previous-versions/

PYG(python 3.9 이상)
torch_geometric:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
Pytorch 버전 및 cuda 버전 확인

torch_sparse, torch_scatter:
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
(Torch-2.6.0+cu126 = PyTorch 2.6.0 and cuda gpu 12.6)

PyTorch lightning
https://lightning.ai/docs/pytorch/stable/ (X)
pytorch-lightning==1.7.7
버전 2.x.x로 올라가면 TSP module 달라져서 문제발생

torchmetrics==0.11.0
(PyTorch lightning과 호환성)

Cython_merge
After installing Cython module using conda,
If we face the following error: 

warning: 
#warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

or

   /* Using NumPy API declarations from "numpy/__init__.cython-30.pxd" */
    
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/arrayscalars.h"
#include "numpy/ufuncobject.h"

Then, “dismiss”.(I’m not sure)
Check implementation “import cython_merge” in the same folder.

아니면
setup.py에서 밑에 extension  안에
define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')] 추가


Installing pyconcorde
qsopt.a, qsopt.h, concorde.c, concorde.h 미리 준비
For linux(PIC?)
https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/
qsopt.PIC.a -> qsopt.a

최신 파일 다운로드
https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.html

terminal에서는
wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
(setup.py 참고)

Do https://github.com/jvkersch/pyconcorde

pip install -e. 했는데 오류 발생하면,
data 폴더 안에 미리 준비한 (qsopt.a, qsopt.h, concorde.c, concorde.h) 전부 넣기


Installing LKH

1. https://pypi.org/project/lkh/ 는 에러 발생
2. Command on cmd terminal:
3. curl -O http://webhotel4.ruc.dk/~keld/research/LKH/LKH-3.0.12.tgz
4. tar -xvzf LKH-3.0.12.tgz
5. cd LKH-3.0.12
6. make
7. sudo cp LKH /usr/local/bin
8.  pip install lkh

1. Example.py

import requests
import lkh

problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
problem = lkh.LKHProblem.parse(problem_str)
print(problem)


solver_path = '../LKH-3.0.12/LKH'
solution = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=10)
tour = [n - 1 for n in solution[0]]

print(f"Tour: {tour}")
print(f"Tour length: {solution[1]}")


2. Example.py

https://github.com/Edward-Sun/DIFUSCO/blob/main/data/generate_tsp_data.py


generate_tsp_data.py #LKH-xx.x.xx relative path and version 수정
lkh_path = "data/LKH-3.0.12/LKH"

from concorde.tsp import TSPSolver
이거를 tsp.py의 relative path로 변경
'''
