set H_PAD=6
set V_PAD=10
set PAD_COMMAND_TOP=pad=iw+%H_PAD%:ih+%V_PAD%:color=white
set PAD_COMMAND_TOP_LAST=pad=iw:ih+%V_PAD%:color=white
set PAD_COMMAND_BOT=pad=iw+%H_PAD%:ih:color=white
set PAD_COMMAND_BOT_LAST=pad=iw:ih:color=white
set FPS=60
set INPUT_DIR=./../code/multiImage_pytorch/out
SET OUTPUT_DIR=./

set INPUT_NAME=train_sample

ffmpeg ^
-loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/i.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_n.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_d.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_r.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_s.jpg ^
-f lavfi -i color=white:s=256x256:r=%FPS%:d=1 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_n_%%d.jpg -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_d_%%d.jpg -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_r_%%d.jpg -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_s_%%d.jpg ^
-filter_complex "[0]%PAD_COMMAND_TOP%[0v];[1]%PAD_COMMAND_TOP%[1v];[2]%PAD_COMMAND_TOP%[2v];[3]%PAD_COMMAND_TOP%[3v];[4]%PAD_COMMAND_TOP_LAST%[4v];[5]%PAD_COMMAND_BOT%[5v];[6]%PAD_COMMAND_BOT%[6v];[7]%PAD_COMMAND_BOT%[7v];[8]%PAD_COMMAND_BOT%[8v];[9]%PAD_COMMAND_BOT_LAST%[9v];[0v][1v][2v][3v][4v]hstack=inputs=5[t];[5v][6v][7v][8v][9v]hstack=inputs=5[b];[t][b]vstack[v]" ^
-map "[v]" -pix_fmt yuv420p %OUTPUT_DIR%/%INPUT_NAME%_temp.mp4

set INPUT_NAME=test_sample

ffmpeg ^
-loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/i.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_n.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_d.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_r.jpg -loop 0 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/target_s.jpg ^
-f lavfi -i color=white:s=256x256:r=%FPS%:d=1 -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_n_%%d.jpg -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_d_%%d.jpg -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_r_%%d.jpg -r %FPS% -i %INPUT_DIR%/%INPUT_NAME%/pred_s_%%d.jpg ^
-filter_complex "[0]%PAD_COMMAND_TOP%[0v];[1]%PAD_COMMAND_TOP%[1v];[2]%PAD_COMMAND_TOP%[2v];[3]%PAD_COMMAND_TOP%[3v];[4]%PAD_COMMAND_TOP_LAST%[4v];[5]%PAD_COMMAND_BOT%[5v];[6]%PAD_COMMAND_BOT%[6v];[7]%PAD_COMMAND_BOT%[7v];[8]%PAD_COMMAND_BOT%[8v];[9]%PAD_COMMAND_BOT_LAST%[9v];[0v][1v][2v][3v][4v]hstack=inputs=5[t];[5v][6v][7v][8v][9v]hstack=inputs=5[b];[t][b]vstack[v]" ^
-map "[v]" -pix_fmt yuv420p %OUTPUT_DIR%/%INPUT_NAME%_temp.mp4

ffmpeg -i %OUTPUT_DIR%/train_sample_temp.mp4 -i %OUTPUT_DIR%test_sample_temp.mp4 -filter_complex "[0]pad=iw:ih+%V_PAD%:color=white[0v];[0v][1v]vstack" %OUTPUT_DIR%/train_test_sample.mp4