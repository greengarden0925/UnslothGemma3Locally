#20250612 Unsloth支援 python=3.10,3.11,3.12.
#建議用conda建立虛擬環境
#用unbuntu terminal 執行
#但是為了讓程式可以執行 xformers比較難符合要求
#python要用3.11
#建立的虛擬環境會在
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=11.8 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

#顯示所虛擬環境
conda info --envs
conda activate unsloth_env
# conda deactivate
#conda remove --name munsloth_env --all #移除虛擬環境 如果需要的話

pip install unsloth


#trainer.train() transformers==4.51.3 原本是4.52.4 chatGPT建議降為==4.51.3
pip uninstall transformers -y #要移除前面的 不然後面會出問題 要降級 
pip install transformers==4.51.3


