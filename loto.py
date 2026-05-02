import pandas as pd
import numpy as np


data_frame = pd.read_csv("loto_649_si_noroc.csv")
# so far vom merge pe BCE pentru ca pot fi mai multe variante simultan
print(data_frame.head())
#feature engineering, scapam de coloana date pentru ca nu ne trebe
data_frame_final = data_frame.drop(columns=['date'])
print(data_frame_final.head())



def encode_single_draw(draw_numbers):
    encoded_vector = [0] * 49

    for number in draw_numbers:
        index = number - 1
        encoded_vector[index] = 1 #sa avem pozitia corecta
    return encoded_vector

first_extraction = data_frame_final.iloc[0].tolist()
print("first extraction: ", first_extraction)

encoded_vector = encode_single_draw(first_extraction)
print("encoded vector: ", encoded_vector)

def encode_all_draws(df_values):
    all_encoded_vector = []

    for row in df_values:
        encoded_row = encode_single_draw(row)
        all_encoded_vector.append(encoded_row)
    return all_encoded_vector

extracted_values = data_frame_final.values
dataset = encode_all_draws(extracted_values)


print("extracted values: ", dataset[:2])

target_x = dataset[:-1]
target_y = dataset[1:]
