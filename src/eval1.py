def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]
def find_category_and_product_v1(user_input, products_and_category):

    delimiter = "####"
    system_message = f"""
        You will be provided with customer service queries. 
        The customer service query will be delimited with {delimiter} characters.
        Output a python list of json objects, where each object has the following format:
            'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
        Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
        AND
            'products': <a list of products that must be found in the allowed products below>
    
    
        Where the categories and products must be found in the customer service query.
        If a product is mentioned, it must be associated with the correct category in the allowed products list below.
        If no products or categories are found, output an empty list.
    
    
        List out all products that are relevant to the customer service query based on how closely it relates
        to the product name and product category.
        Do not assume, from the name of the product, any features or attributes such as relative quality or price.
    
        The allowed products are provided in JSON format.
        The keys of each item represent the category.
        The values of each item is a list of products that are within that category.
        Allowed products: {products_and_category}
    
    
        """

    few_shot_user_1 = """I want the most expensive computer."""
    few_shot_assistant_1 = """ 
        [{'category': 'Computers and Laptops', \
    'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
        """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},
        {'role': 'assistant', 'content': few_shot_assistant_1},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
    ]
    return get_completion_from_messages(messages)