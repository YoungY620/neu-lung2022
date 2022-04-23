# NEU Lung

## 简介

毕业设计：小鼠肺部图片肺炎诊断

## 尝试运行

```powershell
git clone https://github.com/YoungY620/neu-lung2022
cd neu-lung2022
pip install virtualenv
virtualenv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行服务：

```powershell
# in powershell
env:FLASK_ENV = "development"
env:FLASK_APP = "lung"
flask run
```

## 鸣谢

基于以下项目二次开发：

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [SimCLR(Pytorch)](https://github.com/sthalles/SimCLR)
