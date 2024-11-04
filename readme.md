## install
```commandline
pip install -r requirements.txt

```

## run
```commandline
flask --app api/app.py run --port 5009

or
python -m flask --app api/app.py run --port 5009

```

## in aws ec2
```commandline
python -m flask --app api/app.py run --host 0.0.0.0 --port 5009
```