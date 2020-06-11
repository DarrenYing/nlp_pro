import torch


class Controller:
    '''
    模型及参数的加载和保存
    '''

    def __init__(self):
        pass

    def load_model(self, model_path):
        print("load the model")
        try:
            model = torch.load(model_path)
            return model
        except Exception as e:
            print("The model file doesn't exist!")
            exit(1)

    def store_model(self, model, model_path):
        try:
            torch.save(model, model_path)
        except Exception as e:
            print(e)
            exit(1)

    def load_param(self, model, param_path):
        print("load the model")
        try:
            model.load_state_dict(torch.load(param_path))
            return model
        except Exception as e:
            print("The model file doesn't exist!")
            exit(1)

    def store_param(self, model, param_path):
        try:
            torch.save(model.state_dict(), param_path)
        except Exception as e:
            print(e)
            exit(1)





