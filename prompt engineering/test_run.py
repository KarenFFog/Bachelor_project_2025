from gen import *
from eval import generate_responses_for_all_places

def test_run_2():
    # read places from this file
    file_path = "places_.jsonl"

    # write result to this file
    result_file_path = "places_result.jsonl"

    # read prompt from file
    prompt = read_prompt("prompt2.txt")

    # generate responses for all places
    generate_responses_for_all_places(prompt, file_path, result_file_path)

# nearest city test
file_path_geo = "mmearth_coordinates.jsonl"
file_path_prompt = "prompt3.txt"
file_path_result = "nearest_city.jsonl"

places = read_jsonl(file_path_geo)
print(places[0])

prompt = read_prompt(file_path_prompt)
prompt = insert_coordinates(prompt, places[0])
print(prompt)

response = generate_response(prompt)
print(response)

# if __name__ == "__main__":
#     test_run_2()

