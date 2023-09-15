

import shutil
import os

copy_from_base = os.path.join('X:\\', 'George', 'TransferLearning', 'Data', '2023_Feb_Oneiroi')
copy_to_base = os.path.join('D:\\', '2023_Feb_Oneiroi')

rats_folders = ['04_Ikelos', '05_Fovitor', '06_Hypnos', '07_Fantasos', '08_Morfeas', '09_Oneiros']
line_discrim_folder = '2023_07_06-xxx_LineDiscriminate'

for rat_folder in rats_folders:
    from_folder_rat = os.path.join(copy_from_base, rat_folder, line_discrim_folder)
    to_folder_rat = os.path.join(copy_to_base, rat_folder, line_discrim_folder)
    all_dates_folders = os.listdir(from_folder_rat)
    for date_folder in all_dates_folders:
        from_folder = os.path.join(from_folder_rat, date_folder)
        to_folder = os.path.join(to_folder_rat, date_folder)

        if not os.path.isdir(to_folder):
            os.makedirs(to_folder)
            all_files = os.listdir(from_folder)
            for i, file in enumerate(all_files):
                if '.avi' in file:
                    break
            del(all_files[i])

            for file in all_files:
                from_file = os.path.join(from_folder, file)
                to_file = os.path.join(to_folder, file)
                if os.path.isfile(from_file):
                    shutil.copy(from_file, to_folder)
                else:
                    shutil.copytree(from_file, to_file)
            print('Done folder :{}'.format(to_folder))
    print('DONE RAT: {}'.format(rat_folder))
