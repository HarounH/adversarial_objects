# wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
mkdir shapenetcore
unzip -q ShapeNetCore.v2.zip -d shapenetcore
python prepare_shapenet_model.py shapenetcore/ShapeNetCore.v2
