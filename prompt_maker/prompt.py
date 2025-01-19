import json


def make_prompt():
    with open("prompt_maker/data/ravel_city_attribute_to_prompts.json") as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    print(make_prompt())
