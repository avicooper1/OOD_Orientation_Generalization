for i in {0..14}
do
   CUDA_VISIBLE_DEVICES=6 ~/blender -b -noaudio -P paradigm_render.py $i
done