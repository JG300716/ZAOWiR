**For help:**<br>
python main.py -h
<br><br><br>
**To Create calibration.json**<br>
--input <PATH_TO_PICTURES_DIRECTORY> --width <WIDTH> --height <HEIGHT> --size <SIZE [mm]> -json
<br><br><br>
**Example:**<br>
--input J:\Files\Mgr◘\I_Rok\II\ZAOWiR\LAB1\res\s1\Pictures --width 10 --height 7 --size 28.67 -json
<br><br><br>
**To create undistorted pictures:**<br>
--input <PATH_TO_PICTURES_DIRECTORY> -load_json <PATH_TO_CALIBRATION.json_FILE>
<br><br><br>
**Example:**<br>
--input J:\Files\Mgr◘\I_Rok\II\ZAOWiR\LAB1\res\cam4\Pictures -load_json J:\Files\Mgr◘\I_Rok\II\ZAOWiR\LAB1\res\cam4\calibration.json

