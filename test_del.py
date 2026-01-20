from GPT2.dataset_add_numbers import AddNumData
add_num_data = AddNumData(max_num=100)
print(add_num_data.special_chars.get('+', None))
add_num_data[4]
