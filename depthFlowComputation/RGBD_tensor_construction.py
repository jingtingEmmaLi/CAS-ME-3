import cv2
import numpy as np
import scipy.io as io
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

import os

base_dir = '/data/databases/casme3/labeledData'
RGB_dir = f"{base_dir}/RGB"
# depth_dir = f"{base_dir}/depth_inter"
depth_dir = f"{base_dir}/depth"
save_dir = f"{base_dir}/feature_with_depth"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

excel_data = pd.read_excel('./cas(me)3label_JpgIndex_1216.xlsx')
crop_txt = pd.read_csv('./crop info_version2.txt',sep=' |/|crop info:\t',header=None, encoding='utf-8', engine='python')
error_txt = pd.read_csv('./crop_info_local.txt',sep=' |/',header=None, encoding='utf-8', engine='python')
crop_txt.dropna(axis=1, inplace=True)
crop_txt.columns = ['Subject', 'Filename',  'Onset', 'x', 'y', 'w', 'h']
error_txt.columns = ['Subject', 'Filename',  'Onset', 'x', 'y', 'w', 'h']
# same = pd.merge(left=crop_txt, right=error_txt, on=['Subject', 'Filename', 'Onset'])
crop_txt = crop_txt[~crop_txt['w'].isin([0])]
full_crop = crop_txt.append(error_txt)
full_crop.drop(labels=777, inplace=True)

#special_subject = ['spNO.210', 'spNO.211', 'spNO.212', 'spNO.213', 'spNO.214', 'spNO.215']


def save_mat(ro, ra, do, da, x, y, w, h, save_path):
    # crop images of rgb and depth based on the defaut setting
    crop_depth_onset = do[y:y+h, x:x+w]
    crop_depth_apex = da[y:y+h, x:x+w]
    rgb_gray_onset = cv2.cvtColor(ro, cv2.COLOR_BGR2GRAY)
    rgb_gray_apex = cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY)
    crop_rgb_gray_onset = rgb_gray_onset[y:y + h, x:x + w]
    crop_rgb_gray_apex = rgb_gray_apex[y:y + h, x:x + w]

    mean_do = np.mean(crop_depth_onset)
    mean_da = np.mean(crop_depth_apex)

    std_do = np.std(crop_depth_onset)
    std_da = np.std(crop_depth_apex)

    normaldo = crop_depth_onset[(crop_depth_onset >= mean_do - 3 * std_do) * (crop_depth_onset <= mean_do + 3 * std_do)]
    normalda = crop_depth_apex[(crop_depth_apex >= mean_da - 3 * std_da) * (crop_depth_apex <= mean_da + 3 * std_da)]

    print('max value in original crop matrix：', np.max(crop_depth_onset), np.max(crop_depth_apex))
    print('min value in original crop matrix：', np.min(crop_depth_onset), np.min(crop_depth_apex))
    print('non zero min value in original crop matrix：', np.min(crop_depth_onset[crop_depth_onset > 0]), np.min(crop_depth_apex[crop_depth_apex > 0]))
    print('average value in original crop matrix ：', mean_do, mean_da)
    print('std value in original crop matrix：', std_do, std_da)
    # print('max in normalized matrix：', np.max(normaldo), np.max(normalda))
    max_depth = max(np.max(normaldo), np.max(normalda))
    min_depth = min(np.min(crop_depth_onset[crop_depth_onset > 0]), np.min(crop_depth_apex[crop_depth_apex > 0]))

   

    new_max_depth = max(np.max(normaldo[normaldo < min_depth + 400]), np.max(normalda[normalda < min_depth + 400]))
   

    if max_depth < min_depth + 400:
        width = crop_rgb_gray_onset.shape[0]
        height = crop_rgb_gray_onset.shape[1]
        depth = new_max_depth - min_depth + 1
        print(width, height, depth, max_depth, min_depth)
        threed_img_onset = np.zeros((width, height, depth)).astype(np.int16)
        threed_img_apex = np.zeros((width, height, depth)).astype(np.int16)

        for ii in range(width):
            for jj in range(height):
                current_onset_depth = crop_depth_onset[ii, jj]
                current_apex_depth = crop_depth_apex[ii, jj]
                if current_onset_depth <= 0:
                    zz1 = 0
                else:
                    zz1 = current_onset_depth - min_depth
                    if zz1 >= depth:
                        zz1 = 0
                if current_apex_depth <= 0:
                    zz2 = 0
                else:
                    zz2 = current_apex_depth - min_depth
                    if zz2 >= depth:
                        zz2 = 0
                threed_img_onset[ii, jj, zz1] = crop_rgb_gray_onset[ii, jj]
                threed_img_apex[ii, jj, zz2] = crop_rgb_gray_apex[ii, jj]
        print(f'begin to save {save_path}', end='\t')
        io.savemat(save_path, {'onset': threed_img_onset, 'apex': threed_img_apex})




def first():
    # Iterate through the excel

    for idx in range(excel_data.shape[0]):



        current_subject = excel_data.loc[idx]['Subject']
        current_filename = excel_data.loc[idx]['Filename']
        current_onset = excel_data.loc[idx]['Onset']
        current_apex = excel_data.loc[idx]['Apex']
        print('*'*70)
        print(f'current task: {current_subject}/{current_filename}/{current_onset}')

        save_target_path = f'{save_dir}/{current_subject}_{current_filename}_{current_onset}.mat'
        if os.path.exists(save_target_path)==0:

            rgb_onset_path = f'{RGB_dir}/{current_subject}/{current_filename}/color/{current_onset}.jpg'
            rgb_apex_path = f'{RGB_dir}/{current_subject}/{current_filename}/color/{current_apex}.jpg'
            depth_onset_path = f'{depth_dir}/{current_subject}/{current_filename}/depth/{current_onset}.png'
            depth_apex_path = f'{depth_dir}/{current_subject}/{current_filename}/depth/{current_apex}.png'
            # # Handling irregularly named subjects
            # if current_subject in special_subject:
            #     current_subject_alter = current_subject.replace('.', '')
            #     depth_onset_path = f'{depth_dir}/{current_subject_alter}/{current_filename}/depth/{current_onset}.png'
            #     depth_apex_path = f'{depth_dir}/{current_subject_alter}/{current_filename}/depth/{current_apex}.png'

            #
            # First get the path of onset and apex frames' RGB and depth
            depth_onset_path_alter = f'{depth_dir}/{current_subject}/{current_filename}/{current_onset}.png'
            depth_apex_path_alter = f'{depth_dir}/{current_subject}/{current_filename}/{current_apex}.png'

            # If the path information is incorrect, record the log
            if not os.path.exists(rgb_onset_path):
                print(f'{current_subject} {current_filename} {current_onset} RGB-onset path error')
                with open('./depth_feature_error6.txt', 'a') as f:
                    f.write(f'{current_subject} {current_filename} {current_onset} RGB-onset path error\n')
                f.close()
                continue
            if not os.path.exists(rgb_apex_path):
                print(f'{current_subject} {current_filename} {current_onset} RGB-apex path error')
                with open('./depth_feature_error6.txt', 'a') as f:
                    f.write(f'{current_subject} {current_filename} {current_onset} RGB-apex path error\n')
                f.close()
                continue
            if not os.path.exists(depth_onset_path):
                depth_onset_path = depth_onset_path_alter
                if not os.path.exists(depth_onset_path):
                    print(f'{current_subject} {current_filename} {current_onset} depth-onset path error')
                    with open('./depth_feature_error6.txt', 'a') as f:
                        f.write(f'{current_subject} {current_filename} {current_onset} depth-onset path error\n')
                    f.close()
                    continue
            if not os.path.exists(depth_apex_path):
                depth_apex_path = depth_apex_path_alter
                if not os.path.exists(depth_apex_path):
                    print(f'{current_subject} {current_filename} {current_onset} depth-apex path error')
                    with open('./depth_feature_error6.txt', 'a') as f:
                        f.write(f'{current_subject} {current_filename} {current_onset} depth-apex path error\n')
                    f.close()
                    continue

            # If the apex frame is before the onset frame, record a log with the cause of the error
            if current_onset >= current_apex:
                print(f'{current_subject} {current_filename} {current_onset} apex location error')
                with open('./depth_feature_error6', 'a') as f:
                    f.write(f'{current_subject} {current_filename} {current_onset} apex location error\n')
                f.close()
                continue

            # continue

            # If all the information is correct, start getting the information of the crop box
            crop_info = full_crop.loc[(full_crop['Subject']==current_subject) & (full_crop['Filename']==current_filename) & (full_crop['Onset']==current_onset)]
            # print(crop_info)

            crop_x = crop_info['x'].values.item()
            crop_y = crop_info['y'].values.item()
            crop_w = crop_info['w'].values.item()
            crop_h = crop_info['h'].values.item()
            #
            #
            depth_onset = cv2.imread(depth_onset_path, cv2.IMREAD_UNCHANGED)
            depth_apex = cv2.imread(depth_apex_path, cv2.IMREAD_UNCHANGED)
            rgb_onset = cv2.imread(rgb_onset_path)
            rgb_apex = cv2.imread(rgb_apex_path)


            save_mat(rgb_onset, rgb_apex, depth_onset, depth_apex, crop_x, crop_y, crop_w, crop_h, save_target_path)

            with open('./depth_success_info6.txt', 'a') as f:
                f.write(f'{current_subject} {current_filename} {current_onset} {save_target_path}\n')
            f.close()
            print('Save successfully')



def process_error():
    error_info = pd.read_csv('./depth_feature_error5.txt', header=None, sep=' ', encoding='utf-8', engine='python')
    error_info.columns = ['Subject', 'Filename', 'Onset', 'Error_type']
    groups = error_info.groupby('Error_type')
    depth_onset_error = groups.get_group('depth-onset path error').reset_index()
    rgb_apex_error = groups.get_group('apex location error').reset_index()
    rgb_onset_error = groups.get_group('RGB-onset path error').reset_index()

    rgb_onset_excel = pd.merge(left=excel_data, right=rgb_onset_error, on=['Subject', 'Filename', 'Onset'])


    # handling apex location error
    rgb_apex_excel = pd.merge(left=excel_data, right=rgb_apex_error, on=['Subject', 'Filename', 'Onset'])
    # print(rgb_apex_excel.loc[0,'Onset'])
    for i in range(rgb_apex_excel.shape[0]):
        cs = rgb_apex_excel.loc[i, 'Subject']
        cf = rgb_apex_excel.loc[i, 'Filename']
        con = rgb_apex_excel.loc[i, 'Onset']
        cap = rgb_apex_excel.loc[i, 'Apex']
        cof = rgb_apex_excel.loc[i, 'Offset']
        # onset and apex have the same location, the location of apex is set to be the average value

        print('*'*50)
        print(f'current task: {cs}/{cf}/{con}')

        if os.path.exists('./depth_success_info5_1.txt'):
            success = pd.read_csv('./depth_success_info5_1.txt', header=None, sep=' ', engine='python', encoding='utf-8',names=['Subject', 'Filename', 'Onset', 'Target'])
            if not success.loc[(success['Subject'] == cs) & (success['Filename'] == cf) & (success['Onset'] == min(con, cap, cof))].empty:
                print('Completed, start the next sample')
                continue

        alter_apex = -1
        if con == cap:
            alter_apex = (con + cof) // 2
            # Detects the presence of this frame
            alter_apex_path = '{}/{}/{}/color/{}.jpg'.format(RGB_dir, cs, cf, alter_apex)
            # If the top frame does not exist, search backwards for the nearest frame
            while not os.path.exists(alter_apex_path):
                alter_apex += 1
                if alter_apex == cof:
                    break
                alter_apex_path = '{}/{}/{}/color/{}.jpg'.format(RGB_dir, cs, cf, alter_apex)

            # If not found by searching backwards, search forwards
            if alter_apex == cof:
                alter_apex = (con + cof) // 2 -1
                alter_apex_path = '{}/{}/{}/color/{}.jpg'.format(RGB_dir, cs, cf, alter_apex)
                while not os.path.exists(alter_apex_path):
                    alter_apex -= 1
                    if alter_apex == con:
                        break
                    alter_apex_path = '{}/{}/{}/color/{}.jpg'.format(RGB_dir, cs, cf, alter_apex)

            # Determine if the current frame is between the onset frame and the offset frame
            if not con < alter_apex < cof:
                with open('./depth_error_5_1.txt', 'a') as f:
                    print('{}/{}/{} apex error'.format(cs, cf, con))
                    f.write('{}/{}/{} apex error\n'.format(cs, cf, con))
                f.close()
                continue

        if con > cap:
            onset_apex_offsex = [con, cap, cof]
            onset_apex_offsex.sort()
            con = onset_apex_offsex[0]
            alter_apex = onset_apex_offsex[1]

        rgb_on_path = f'{RGB_dir}/{cs}/{cf}/color/{con}.jpg'
        rgb_ap_path = f'{RGB_dir}/{cs}/{cf}/color/{alter_apex}.jpg'
        depth_on_path = f'{depth_dir}/{cs}/{cf}/depth/{con}.png'
        depth_ap_path = f'{depth_dir}/{cs}/{cf}/depth/{cap}.png'
        # Handling irregularly named subjects
        # if cs in special_subject:
        #     current_subject_alter = cs.replace('.', '')
        #     depth_on_path = f'{depth_dir}/{current_subject_alter}/{cf}/depth/{con}.png'
        #     depth_ap_path = f'{depth_dir}/{current_subject_alter}/{cf}/depth/{cap}.png'

        depth_on_path_alter = f'{depth_dir}/{cs}/{cf}/{con}.png'
        depth_ap_path_alter = f'{depth_dir}/{cs}/{cf}/{cap}.png'


        # If the path information is incorrect, record the log
        if not os.path.exists(rgb_on_path):
            print(f'{cs} {cf} {con} RGB-onset path error')
            with open('./depth_error_5_1.txt', 'a') as f:
                f.write(f'{cs} {cf} {con} RGB-onset path error\n')
            f.close()
            continue
        if not os.path.exists(rgb_ap_path):
            print(f'{cs} {cf} {con} RGB-apex path error')
            with open('./depth_error_5_1.txt', 'a') as f:
                f.write(f'{cs} {cf} {con} RGB-apex path error\n')
            f.close()
            continue
        if not os.path.exists(depth_on_path):
            depth_on_path = depth_on_path_alter
            if not os.path.exists(depth_on_path):
                print(f'{cs} {cf} {con} depth-onset path error')
                with open('./depth_error_5_1.txt', 'a') as f:
                    f.write(f'{cs} {cf} {con} depth-onset path error\n')
                f.close()
                continue
        if not os.path.exists(depth_ap_path):
            depth_ap_path = depth_ap_path_alter
            if not os.path.exists(depth_ap_path):
                print(f'{cs} {cf} {con} depth-apex location error')
                with open('./depth_error_5_1.txt', 'a') as f:
                    f.write(f'{cs} {cf} {con} depth-apex location error\n')
                f.close()
                continue

        crop_info = full_crop.loc[(full_crop['Subject']==cs) & (full_crop['Filename']==cf) & (full_crop['Onset']==con)]
        crop_x = crop_info['x'].values.item()
        crop_y = crop_info['y'].values.item()
        crop_w = crop_info['w'].values.item()
        crop_h = crop_info['h'].values.item()
        #
        #
        depth_onset = cv2.imread(depth_on_path, cv2.IMREAD_UNCHANGED)
        depth_apex = cv2.imread(depth_ap_path, cv2.IMREAD_UNCHANGED)
        rgb_onset = cv2.imread(rgb_on_path)
        rgb_apex = cv2.imread(rgb_ap_path)

        save_target_path = f'{save_dir}/{cs}_{cf}_{con}.mat'
        save_mat(rgb_onset, rgb_apex, depth_onset, depth_apex, crop_x, crop_y, crop_w, crop_h, save_target_path)
        with open('./depth_success_info6.txt', 'a') as f:
            f.write(f'{cs} {cf} {con} {save_target_path}\n')
        f.close()
        print('Save successfully')


if __name__ == '__main__':
    #first()
    #process_error()
    # Handle the case of spNO.216 onset and apex frame is 0, set the onset frame to 1 and the apex frame to 12
    ro_path = f'{RGB_dir}/spNO.216/e/color/1.jpg'
    ra_path = f'{RGB_dir}/spNO.216/e/color/12.jpg'
    do_path = f'{depth_dir}/spNO.216/e/1.png'
    da_path = f'{depth_dir}/spNO.216/e/12.png'

    ro = cv2.imread(ro_path)
    ra = cv2.imread(ra_path)
    do = cv2.imread(do_path, cv2.IMREAD_UNCHANGED)
    da = cv2.imread(da_path, cv2.IMREAD_UNCHANGED)
    sx, sy, sw, sh = 557, 379, 268, 268
    sp = f'{save_dir}/spNO.216_e_0.mat'
    save_mat(ro, ra, do, da, sx, sy, sw, sh, sp)
    with open('./depth_success_info6.txt', 'a') as f:
        f.write(f'spNO.216 e 0 {sp}\n')
    f.close()
    print('Save successfully')



