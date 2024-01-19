# NYCU_Data_Mining_Final_Project

## Usage

```
python main.py -d %dataset dir% -m %recommand method% -k %nearest neighbor num% -r %recommend num%
```

### example
```
python main.py -d ../dataset -m kmeans -k 20 -r 10
python main.py -d ../dataset -m user_based_cf -k 20 -r 10
```