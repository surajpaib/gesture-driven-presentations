#!/bin/bash
#First flip all
python mirror.py

#Move the flipped ones into the correct folder
cd ../right_arm_prev_flipped

for v1 in *flipped.mp4
do
mv $v1 ../left_arm_next
done

cd ../right_arm_next_flipped

for v1 in *flipped.mp4
do
mv $v1 ../left_arm_prev
done

cd ../left_arm_prev_flipped

for v1 in *flipped.mp4
do
mv $v1 ../right_arm_next
done

cd ../left_arm_next_flipped

for v1 in *flipped.mp4
do
mv $v1 ../right_arm_prev
done

cd ../reset_flipped

for v1 in *flipped.mp4
do
mv $v1 ../reset
done

cd ../start_stop_flipped

for v1 in *flipped.mp4
do
mv $v1 ../start_stop
done

#Reverse all

cd right_arm_next
for v in *.mp4
do
ffmpeg -i $v -vf reverse -af areverse $v-reversed.mp4
done

for v1 in *reversed.mp4
do
mv $v1 ../right_arm_prev_reversed
done

echo "right_arm_next videos converted into right_arm_prev"

cd ../right_arm_prev 

for v in *.mp4
do
ffmpeg -i $v -vf reverse -af areverse $v-reversed.mp4
done

for v1 in *reversed.mp4
do
mv $v1 ../right_arm_next_reversed
done

echo "right_arm_prev videos converted into right_arm_next"

cd ../left_arm_next

for v in *.mp4
do
ffmpeg -i $v -vf reverse -af areverse $v-reversed.mp4
done

for v1 in *reversed.mp4
do
mv $v1 ../left_arm_prev_reversed
done

echo ""

cd ../left_arm_prev

for v in *.mp4
do
ffmpeg -i $v -vf reverse -af areverse $v-reversed.mp4
done

for v1 in *reversed.mp4
do
mv $v1 ../left_arm_next_reversed
done

cd ../start_stop

for v in *.mp4
do
ffmpeg -i $v -vf reverse -af areverse $v-reversed.mp4
done

cd ../reset

for v in *.mp4
do
ffmpeg -i $v -vf reverse -af areverse $v-reversed.mp4
done

#Move files from the reversed folder into the correct one
cd ../right_arm_prev_reversed

for v1 in *reversed.mp4
do
mv $v1 ../right_arm_prev
done

cd ../right_arm_next_reversed

for v1 in *reversed.mp4
do
mv $v1 ../right_arm_next
done

cd ../left_arm_prev_reversed

for v1 in *reversed.mp4
do
mv $v1 ../left_arm_prev
done

cd ../left_arm_next_reversed

for v1 in *reversed.mp4
do
mv $v1 ../left_arm_next
done

