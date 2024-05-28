import pandas as pd


def transform_data(data_path: str, save_file_path: str):
    df.dropna(inplace=True)
    df.columns = df.columns.str.lower()

    df["journey_day"] = pd.to_datetime(df["date_of_journey"], format="%d/%m/%Y").dt.day
    df["journey_month"] = pd.to_datetime(
        df["date_of_journey"], format="%d/%m/%Y"
    ).dt.month
    df.drop("date_of_journey", axis=1, inplace=True)

    df["dep_hour"] = pd.to_datetime(df["dep_time"]).dt.hour
    df["dep_minutes"] = pd.to_datetime(df["dep_time"]).dt.minute
    df.drop("dep_time", axis=1, inplace=True)

    df["arrival_hour"] = pd.to_datetime(df["arrival_time"]).dt.hour
    df["arrival_minutes"] = pd.to_datetime(df["arrival_time"]).dt.minute
    df.drop("arrival_time", axis=1, inplace=True)

    df[["duration_hours", "duration_minutes"]] = df["duration"].str.extract(
        r"(\d+h)?\s?(\d+m|\d+)?", expand=True
    )
    df["duration_hours"] = (
        df["duration_hours"].str.strip("h").astype(float).fillna(0).astype(int)
    )
    df["duration_minutes"] = (
        df["duration_minutes"].str.strip("m").astype(float).fillna(0).astype(int)
    )

    df.drop(["duration"], axis=1, inplace=True)

    airline = df[["airline"]]
    airline = pd.get_dummies(airline, drop_first=True)

    source = df[["source"]]
    source = pd.get_dummies(source, drop_first=True)

    destination = df[["destination"]]
    destination = pd.get_dummies(destination, drop_first=True)

    df.drop(["route", "additional_info"], axis=1, inplace=True)

    total_stops = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}

    df["total_stops"] = df["total_stops"].map(total_stops).astype("int")

    df = pd.concat([df, airline, source, destination], axis=1)

    df.drop(["airline", "source", "destination"], axis=1, inplace=True)

    df.to_csv(save_file_path, index=False)
