import pandas as pd
import yaml

flat_data = pd.read_csv("data/cars-1k.csv")
nested_data = []

for row in flat_data.to_dict("records"):
    nested_data.append(
        {
            "price": row["price"],
            "year": row["year"],
            "model": {
                "make": row["make"],
                "manufacturer": row["manufacturer"],
            },
            "miles": row["miles"],
            "title_status": row["title_status"],
            "engine": {
                "fuel": row["fuel"],
                "transmission": row["transmission"],
                "drive": row["drive"],
            },
            "type": row["type"],
            "paint_color": row["paint_color"],
        }
    )

with open("data/cars-1k.yaml", "w") as fp:
    yaml.dump(nested_data, fp)
