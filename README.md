执行方法：
python im2rec.py --list=1 --recursive=1 --shuffle=1 --test_ratio=0.2 --train_ratio=0.8 ./pig ./extracted_images/
python im2rec.py --num_thread=4 --pass-through=1 ./pig ./extracted_images/
./extracted_images/就是我们生成的猪图片的路径
