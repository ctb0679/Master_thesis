import ollama

response = ollama.chat(
    model='qwen3.6:27b',
    messages=[{
        'role': 'user',
        'content': 'Describe this image in detail:',
        'images': ['/home/idac/Junaidali/Master_thesis/test.png']
    }]
)

print(response['message']['content'])
