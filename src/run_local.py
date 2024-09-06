from langchain_community.llms import CTransformers


def initialize_llm():
    

    llm = CTransformers(model= "model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type='llama',
                        config={'max_new_tokens': 128,
                                'temperature': 0.5,
                                "context_length": 5000}
                                )
    return llm



if __name__ == '__main__':
    llm = initialize_llm()
    print(llm.invoke("What is Angular momentum??"))




