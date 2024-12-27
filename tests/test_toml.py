import cbrkit

data = cbrkit.loaders.file("data/cars-1k.yaml")
cbrkit.dumpers.file(data, "data/cars-1k.toml")
data2 = cbrkit.loaders.file("data/cars-1k.toml")
print(data2)
