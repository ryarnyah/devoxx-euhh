# How TO
* Prepare build env (with virtualenv)
```sh
virtualenv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```
* Add your audio data in `final-data` folder (Euuh & Yolo)
* Process files & train model
```sh
python train.py
```
* Model will be available in tfjs-model
