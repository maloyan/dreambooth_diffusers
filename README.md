# dreambooth from diffusers library example

This example shows how to use the diffusers library to create a dreambooth images.

## Usage

### Config 

look at config.yaml
    * change your name in ```class_prompt``` and  ```instance promt``` 
    * put your photos in folder and set path in ```instance_data_dir```
    
### Run 
```
pip install -e .
python dreambooth_tutorial/train.py
```