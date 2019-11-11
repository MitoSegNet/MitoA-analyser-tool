"""

Class Control
Contains all functions that are necessary for the entire program

Class MidWindow
Opens additional windows to reach deeper functionalities of Analyser

Class GetMeasurements
Collects shape descriptors and saves them to excel file

Class MorphComparison
Perform statistical analysis and generate boxplots of two samples

Class CorrelationAnalysis
Check correlation between up to 4 descriptors of max two-samples

Class MultiMorphComparison
Perform statistical analysis and generate boxplots of multiple samples



"""

from tkinter import *
import tkinter.font
import tkinter.messagebox
import tkinter.filedialog
import os
import webbrowser
import collections
import copy
import itertools

import numpy as np
import cv2
import pandas as pd
from skimage import img_as_bool
from skimage.morphology import skeletonize
from skan import summarise
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import normaltest, mannwhitneyu, ttest_ind, f_oneway, kruskal, levene, pearsonr, spearmanr
from Plot_Significance import significance_bar

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", DeprecationWarning)


# pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)


# GUI
####################

class Control:
    """
    Adds functions that can be accessed from all windows in the program
    """

    def __init__(self):
        pass

    # close currently open window
    def close_window(self, window):

        window.destroy()

    # opens link to documentation of how to use the program
    def help(self, window):

        webbrowser.open_new("https://github.com/MitoSegNet/MitoSegNet_Analyser")

    # open new window with specified width and height
    def new_window(self, window, title, width, height):

        window.title(title)

        window.minsize(width=int(width / 2), height=int(height / 2))
        window.geometry(str(width) + "x" + str(height) + "+0+0")

    # adds menu to every window, which contains the above functions close_window, help and go_back
    def small_menu(self, window):

        menu = Menu(window)
        window.config(menu=menu)

        submenu = Menu(menu)
        menu.add_cascade(label="Menu", menu=submenu)

        submenu.add_command(label="Help", command=lambda: self.help(window))

        # creates line to separate group items
        submenu.add_separator()

        submenu.add_command(label="Go Back", command=lambda: self.go_back(window, root))
        submenu.add_command(label="Exit", command=lambda: self.close_window(window))

    def place_text(self, window, text, x, y, height, width):

        if height is None or width is None:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y)
        else:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_button(self, window, text, func, x, y, height, width):

        Button(window, text=text, command=func).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_entry(self, window, text, x, y, height, width):

        Entry(window, textvariable=text).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def stat_and_desc(self):

        descriptor_list = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter", "Solidity",
                           "Mean Intensity", "Max Intensity", "Min Intensity", "Number of branches", "Branch length",
                           "Total branch length", "Curvature index"]

        stat_list = ["Average", "Median", "Standard Deviation", "Standard Error", "Minimum", "Maximum", "N"]

        return descriptor_list, stat_list


class MidWindow(Control):

    def __init__(self):
        Control.__init__(self)

    def finalwindow1(self):
        fw1 = Tk()

        self.new_window(fw1, "MitoSegNet Analyser - Analysis of 2 samples", 300, 270)
        self.small_menu(fw1)

        morph_comparison = MorphComparison()
        correlation = CorrelationAnalysis()

        control_class.place_button(fw1, "Morphological comparison\nGenerate table",
                                   morph_comparison.morph_comparison_table, 85, 20, 60, 150)
        control_class.place_button(fw1, "Morphological comparison\nGenerate plots",
                                   morph_comparison.morph_comparison_plot, 85, 100, 60, 150)
        control_class.place_button(fw1, "Correlation analysis", correlation.corr_analysis, 85, 180, 60, 150)

    def finalwindow2(self):
        fw2 = Tk()

        multi_morph_comparison = MultiMorphComparison()

        self.new_window(fw2, "MitoSegNet Analyser - Analysis of multiple samples", 300, 190)
        self.small_menu(fw2)

        control_class.place_button(fw2, "Morphological comparison\nGenerate table",
                                   multi_morph_comparison.morph_comparison_table, 85, 20, 60, 150)
        control_class.place_button(fw2, "Morphological comparison\nGenerate plots",
                                   multi_morph_comparison.morph_comparison_plot, 85, 100, 60, 150)

    def midwindow(self):
        mw = Tk()

        self.new_window(mw, "MitoSegNet Analyser - Analysis", 300, 190)
        self.small_menu(mw)

        control_class.place_button(mw, "2 samples", self.finalwindow1, 85, 20, 60, 150)
        control_class.place_button(mw, "More than 2 samples", self.finalwindow2, 85, 100, 60, 150)


class Get_Measurements(Control):

    def __init__(self):
        Control.__init__(self)

    def get_measurements_window(self):

        gm_root = Tk()

        self.new_window(gm_root, "MitoSegNet Analyser - Get measurements", 500, 380)
        self.small_menu(gm_root)

        table_name = StringVar(gm_root)
        imgpath = StringVar(gm_root)
        labpath = StringVar(gm_root)
        dirpath = StringVar(gm_root)

        def askopenimgs():
            set_imgpath = tkinter.filedialog.askdirectory(parent=gm_root, title='Choose a directory')
            imgpath.set(set_imgpath)

        def askopenlabels():
            set_labpath = tkinter.filedialog.askdirectory(parent=gm_root, title='Choose a directory')
            labpath.set(set_labpath)

        def askopendir():
            set_dirpath = tkinter.filedialog.askdirectory(parent=gm_root, title='Choose a directory')
            dirpath.set(set_dirpath)

        #### enter table name

        self.place_text(gm_root, "Enter name of measurements table", 15, 20, None, None)
        self.place_entry(gm_root, table_name, 25, 50, 30, 400)

        #### browse for table saving location

        self.place_text(gm_root, "Select directory in which table should be saved", 15, 90, None, None)
        self.place_button(gm_root, "Browse", askopendir, 435, 120, 30, 50)
        self.place_entry(gm_root, dirpath, 25, 120, 30, 400)

        #### browse for raw image data

        self.place_text(gm_root, "Select directory in which 8-bit raw images are stored", 15, 160, None, None)
        self.place_button(gm_root, "Browse", askopenimgs, 435, 190, 30, 50)
        self.place_entry(gm_root, imgpath, 25, 190, 30, 400)

        #### browse for labels

        self.place_text(gm_root, "Select directory in which segmented images are stored", 15, 230, None, None)
        self.place_button(gm_root, "Browse", askopenlabels, 435, 260, 30, 50)
        self.place_entry(gm_root, labpath, 25, 260, 30, 400)

        def measure():

            img_list = os.listdir(imgpath.get())

            dataframe = pd.DataFrame(columns=["Image", "Measurement", "Average", "Median", "Standard Deviation",
                                              "Standard Error", "Minimum", "Maximum", "N"])

            dataframe_branch = copy.copy(dataframe)

            n = 0
            n2 = 0

            print("Measuring images ...\n")

            if img_list == os.listdir(labpath.get()):

                for i, img in enumerate(img_list):

                    print(i, img)

                    read_img = cv2.imread(imgpath.get() + os.sep + img, -1)
                    read_lab = cv2.imread(labpath.get() + os.sep + img, cv2.IMREAD_GRAYSCALE)

                    # skeletonize
                    ##########################

                    """    
                    05-07-19

                    for some reason the sumarise function of skan prints out a different number of objects, which is why
                    i currently cannot include the branch data in the same table as the morph parameters    
                    """

                    # read_lab_skel = img_as_bool(color.rgb2gray(io.imread(labpath.get() + os.sep + img)))
                    read_lab_skel = img_as_bool(cv2.imread(labpath.get() + os.sep + img, cv2.IMREAD_GRAYSCALE))
                    lab_skel = skeletonize(read_lab_skel).astype("uint8")

                    branch_data = summarise(lab_skel)

                    curve_ind = []
                    for bd, ed in zip(branch_data["branch-distance"], branch_data["euclidean-distance"]):

                        if ed != 0.0:
                            curve_ind.append((bd - ed) / ed)
                        else:
                            curve_ind.append(bd - ed)

                    branch_data["curvature-index"] = curve_ind

                    grouped_branch_data_mean = branch_data.groupby(["skeleton-id"], as_index=False).mean()

                    grouped_branch_data_sum = branch_data.groupby(["skeleton-id"], as_index=False).sum()

                    counter = collections.Counter(branch_data["skeleton-id"])

                    n_branches = []
                    for i in grouped_branch_data_mean["skeleton-id"]:
                        n_branches.append(counter[i])

                    branch_len = grouped_branch_data_mean["branch-distance"].tolist()
                    tot_branch_len = grouped_branch_data_sum["branch-distance"].tolist()

                    curv_ind = grouped_branch_data_mean["curvature-index"].tolist()

                    ##########################

                    labelled_img = label(read_lab)

                    labelled_img_props = regionprops(label_image=labelled_img, intensity_image=read_img,
                                                     coordinates='xy')

                    area = [obj.area for obj in labelled_img_props]
                    minor_axis_length = [obj.minor_axis_length for obj in labelled_img_props]
                    major_axis_length = [obj.major_axis_length for obj in labelled_img_props]
                    eccentricity = [obj.eccentricity for obj in labelled_img_props]
                    perimeter = [obj.perimeter for obj in labelled_img_props]
                    solidity = [obj.solidity for obj in labelled_img_props]
                    mean_int = [obj.mean_intensity for obj in labelled_img_props]
                    max_int = [obj.max_intensity for obj in labelled_img_props]
                    min_int = [obj.min_intensity for obj in labelled_img_props]

                    def add_to_dataframe(df, measure_str, measure, n):

                        df.loc[n] = [img] + [measure_str, np.average(measure), np.median(measure), np.std(measure),
                                             np.std(measure) / np.sqrt(len(measure)), np.min(measure), np.max(measure),
                                             len(measure)]

                    meas_str_l = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter",
                                  "Solidity",
                                  "Mean Intensity", "Max Intensity", "Min Intensity"]
                    meas_l = [area, minor_axis_length, major_axis_length, eccentricity, perimeter, solidity, mean_int,
                              max_int,
                              min_int]

                    #########

                    meas_str_l_branch = ["Number of branches", "Branch length", "Total branch length",
                                         "Curvature index"]
                    meas_l_branch = [n_branches, branch_len, tot_branch_len, curv_ind]

                    #########

                    for m_str, m in zip(meas_str_l, meas_l):
                        add_to_dataframe(dataframe, m_str, m, n)
                        n += 1

                    for m_str_b, mb in zip(meas_str_l_branch, meas_l_branch):
                        add_to_dataframe(dataframe_branch, m_str_b, mb, n2)
                        n2 += 1

                writer = pd.ExcelWriter(dirpath.get() + os.sep + table_name.get() + "_MorphMeasurements_Table.xlsx",
                                        engine='xlsxwriter')

                dataframe.to_excel(writer, sheet_name="ShapeDescriptors")
                dataframe_branch.to_excel(writer, sheet_name="BranchAnalysis")

                writer.save()

                tkinter.messagebox.showinfo("Done", "Table generated", parent=gm_root)

            else:
                tkinter.messagebox.showinfo("Error", "Image names in raw and segmentation folder are not identical",
                                            parent=gm_root)

        self.place_button(gm_root, "Get Measurements", measure, 200, 330, 30, 110)
        gm_root.mainloop()


class MorphComparison(Control):

    def __init__(self):
        Control.__init__(self)

    def get_stats(self, desc, table_path1, table_path2, tab1_name, tab2_name, stat_val):

        if desc == "Number of branches" or desc == "Branch length" or desc == "Total branch length" or desc == "Curvature index":

            table1 = pd.read_excel(table_path1, sheet_name="BranchAnalysis")
            table2 = pd.read_excel(table_path2, sheet_name="BranchAnalysis")

        else:

            table1 = pd.read_excel(table_path1, sheet_name="ShapeDescriptors")
            table2 = pd.read_excel(table_path2, sheet_name="ShapeDescriptors")

        data_d = {}
        max_vals = []
        normtest_list = []

        norm_p = []

        for table, table_name in zip([table1, table2], [tab1_name, tab2_name]):

            # how to acess measurements
            meas_table = table[table["Measurement"] == desc]

            # how to acess statistical values
            values_list = meas_table[stat_val].tolist()

            data_d.update({table_name: values_list})

            max_vals.append(np.max(values_list))

            if len(values_list) >= 8:

                if normaltest(values_list)[1] > 0.05:
                    normtest_list.append(True)
                else:
                    normtest_list.append(False)

                norm_p.append(normaltest(values_list)[1])

        # converting dictionary with different list lengths into a pandas dataframe
        dataframe = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_d.items()]))

        def compare_samples(stat_func):

            data1_l = []
            data2_l = []
            pval_l = []
            eff_siz_l = []

            for a, b in itertools.combinations([tab1_name, tab2_name], 2):

                if dataframe[a].dropna().tolist() != dataframe[b].dropna().tolist():

                    pval = stat_func(dataframe[a].dropna(), dataframe[b].dropna())
                    eff_siz = cohens_d(dataframe[a].dropna(), dataframe[b].dropna())

                else:
                    pval = [1, 1]
                    eff_siz = 0

                data1_l.append(a)
                data2_l.append(b)
                pval_l.append(pval[1])
                eff_siz_l.append(eff_siz)

            return data1_l, data2_l, pval_l, eff_siz_l

        # pooled standard deviation for calculation of effect size (cohen's d)
        def cohens_d(data1, data2):

            p_std = np.sqrt(((len(data1) - 1) * np.var(data1) + (len(data2) - 1) * np.var(data2)) / (
                    len(data1) + len(data2) - 2))

            cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

            return cohens_d

        if len(values_list) >= 8:

            if False in normtest_list:

                data1_l, data2_l, pval_l, eff_siz_l = compare_samples(mannwhitneyu)

                hyp_test = "Mann-Whitney U test"

            else:
                data1_l, data2_l, pval_l, eff_siz_l = compare_samples(ttest_ind)

                hyp_test = "T-test (two independent samples)"

            return norm_p, dataframe, pval_l, eff_siz_l, hyp_test, max_vals

        else:

            return False, dataframe, False, False, False, False

    def morph_comparison_table(self):

        ma_root = Tk()

        self.new_window(ma_root, "MitoSegNet Analyser - Get statistics of morphological comparison", 500, 440)
        self.small_menu(ma_root)

        table_path1 = StringVar(ma_root)
        table1_name = StringVar(ma_root)
        table_path2 = StringVar(ma_root)
        table2_name = StringVar(ma_root)
        stat_value = StringVar(ma_root)

        def askopentable1():
            set_tablepath1 = tkinter.filedialog.askopenfilename(parent=ma_root, title='Select file')
            table_path1.set(set_tablepath1)

        def askopentable2():
            set_tablepath2 = tkinter.filedialog.askopenfilename(parent=ma_root, title='Select file')
            table_path2.set(set_tablepath2)

        #### browse for first table location

        self.place_text(ma_root, "Select measurements table file 1", 15, 20, None, None)
        self.place_button(ma_root, "Browse", askopentable1, 435, 50, 30, 50)
        self.place_entry(ma_root, table_path1, 25, 50, 30, 400)

        #### enter table 1 name

        self.place_text(ma_root, "Enter name of sample 1 (e. g. wild type)", 15, 90, None, None)
        self.place_entry(ma_root, table1_name, 25, 110, 30, 400)

        #### browse for second table location (to compare against table 1)

        self.place_text(ma_root, "Select measurements table file 2 (to compare against table 1)", 15, 160, None, None)
        self.place_button(ma_root, "Browse", askopentable2, 435, 190, 30, 50)
        self.place_entry(ma_root, table_path2, 25, 190, 30, 400)

        #### enter table 2 name

        self.place_text(ma_root, "Enter name of sample 2 (e. g. mutant)", 15, 230, None, None)
        self.place_entry(ma_root, table2_name, 25, 250, 30, 400)

        descriptor_list, stat_list = control_class.stat_and_desc()

        stat_value.set("Average")
        Label(ma_root, text="Select statistical value to analyse", bd=1).place(bordermode=OUTSIDE, x=15, y=300)
        popupMenu_stat = OptionMenu(ma_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=25, y=320, height=30, width=160)

        def start_analysis():

            stat_val = stat_value.get()

            tab1_name = table1_name.get()
            tab2_name = table2_name.get()

            new_tab1_name = tab1_name + " normality test p-value"
            new_tab2_name = tab2_name + " normality test p-value"

            stat_frame = pd.DataFrame(columns=["Descriptor", new_tab1_name, new_tab2_name, "Hypothesis test",
                                               "Hypothesis test p-value", "Effect size", "N"])

            norm_p_l1 = []
            norm_p_l2 = []
            hyp_t_l = []
            hyp_p_l = []
            es_l = []
            n_l = []

            parent_path = os.path.dirname(table_path1.get())

            writer = pd.ExcelWriter(parent_path + os.sep + stat_val + "_Stat_Table.xlsx", engine='xlsxwriter')
            stat_frame.to_excel(writer, sheet_name="Statistics_Summary")

            for desc in descriptor_list:

                norm_p, dataframe, pval_l, eff_siz_l, hyp_test, max_vals = self.get_stats(desc, table_path1.get(),
                                                                                          table_path2.get(), tab1_name,
                                                                                          tab2_name, stat_val)

                if norm_p == False:
                    raise ValueError('Number of samples needs to be 8 or higher for statistical analysis')

                dataframe.to_excel(writer, sheet_name="sing_vals_" + desc)

                norm_p_l1.append(norm_p[0])
                norm_p_l2.append(norm_p[1])
                hyp_t_l.append(hyp_test)
                hyp_p_l.append(pval_l[0])
                es_l.append(eff_siz_l[0])
                n_l.append(len(dataframe))

            stat_frame["Descriptor"] = descriptor_list
            stat_frame[new_tab1_name] = norm_p_l1
            stat_frame[new_tab2_name] = norm_p_l2
            stat_frame["Hypothesis test"] = hyp_t_l
            stat_frame["Hypothesis test p-value"] = hyp_p_l
            stat_frame["Effect size"] = es_l
            stat_frame["N"] = n_l

            stat_frame.to_excel(writer, sheet_name="Statistics_Summary")

            writer.save()

            tkinter.messagebox.showinfo("Done", "Table generated", parent=ma_root)

        self.place_button(ma_root, "Create table", start_analysis, 195, 370, 30, 110)

        ma_root.mainloop()

    def morph_comparison_plot(self):

        ma_root = Tk()

        self.new_window(ma_root, "MitoSegNet Analyser - Plot morphological comparison", 500, 450)
        self.small_menu(ma_root)

        table_path1 = StringVar(ma_root)
        table1_name = StringVar(ma_root)
        table_path2 = StringVar(ma_root)
        table2_name = StringVar(ma_root)
        descriptor = StringVar(ma_root)
        stat_value = StringVar(ma_root)

        def askopentable1():
            set_tablepath1 = tkinter.filedialog.askopenfilename(parent=ma_root, title='Select file')
            table_path1.set(set_tablepath1)

        def askopentable2():
            set_tablepath2 = tkinter.filedialog.askopenfilename(parent=ma_root, title='Select file')
            table_path2.set(set_tablepath2)

        #### browse for first table location

        self.place_text(ma_root, "Select measurements table file 1", 15, 20, None, None)
        self.place_button(ma_root, "Browse", askopentable1, 435, 50, 30, 50)
        self.place_entry(ma_root, table_path1, 25, 50, 30, 400)

        #### enter table 1 name

        self.place_text(ma_root, "Enter name of table 1 (e. g. wild type)", 15, 90, None, None)
        self.place_entry(ma_root, table1_name, 25, 110, 30, 400)

        #### browse for second table location (to compare against table 1)

        self.place_text(ma_root, "Select measurements table file 2 (to compare against table 1)", 15, 160, None, None)
        self.place_button(ma_root, "Browse", askopentable2, 435, 190, 30, 50)
        self.place_entry(ma_root, table_path2, 25, 190, 30, 400)

        #### enter table 2 name

        self.place_text(ma_root, "Enter name of table 2 (e. g. mutant)", 15, 230, None, None)
        self.place_entry(ma_root, table2_name, 25, 250, 30, 400)

        descriptor_list, stat_list = control_class.stat_and_desc()

        descriptor.set("Area")
        Label(ma_root, text="Select shape descriptor to display", bd=1).place(bordermode=OUTSIDE, x=15, y=300)
        popupMenu_desc = OptionMenu(ma_root, descriptor, *set(descriptor_list))
        popupMenu_desc.place(bordermode=OUTSIDE, x=25, y=320, height=30, width=160)

        stat_value.set("Average")
        Label(ma_root, text="Select statistical value to display", bd=1).place(bordermode=OUTSIDE, x=255, y=300)
        popupMenu_stat = OptionMenu(ma_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=265, y=320, height=30, width=160)

        def start_analysis():

            desc = descriptor.get()
            stat_val = stat_value.get()

            tab1_name = table1_name.get()
            tab2_name = table2_name.get()

            ylab = stat_val + " " + desc.lower()
            ylab_size = 34

            xlab = [table1_name.get(), table2_name.get()]

            new_tab1_name = tab1_name + " normality test p-value"
            new_tab2_name = tab2_name + " normality test p-value"

            stat_frame = pd.DataFrame(
                columns=[new_tab1_name, new_tab2_name, "Hypothesis test", "Hypothesis test p-value",
                         "Effect size", "N"])

            norm_p, dataframe, pval_l, eff_siz_l, hyp_test, max_vals = self.get_stats(desc, table_path1.get(),
                                                                                      table_path2.get(),
                                                                                      tab1_name, tab2_name, stat_val)

            if norm_p != False:

                # table with p-values and effect sizes
                ########
                stat_frame[new_tab1_name] = [norm_p[0]]
                stat_frame[new_tab2_name] = [norm_p[1]]
                stat_frame["Hypothesis test"] = [hyp_test]
                stat_frame["Hypothesis test p-value"] = pval_l
                stat_frame["Effect size"] = eff_siz_l
                stat_frame["N"] = [len(dataframe)]
                ########

                increase = 0
                for index, row in stat_frame.iterrows():

                    if row["Hypothesis test p-value"] > 0.05:
                        p = 0

                    elif 0.01 < row["Hypothesis test p-value"] < 0.05:
                        p = 1

                    elif 0.001 < row["Hypothesis test p-value"] < 0.01:
                        p = 2

                    else:
                        p = 3

                    max_bar = np.max(max_vals)

                    x1 = 0
                    x2 = 1

                    significance_bar(pos_y=max_bar + 0.1 * max_bar + increase, pos_x=[x1, x2], bar_y=max_bar * 0.05,
                                     p=p,
                                     y_dist=max_bar * 0.02,
                                     distance=0.05)

                    increase += max_bar * 0.1

            # select plot
            plot = sb.boxplot(data=dataframe, color="white", fliersize=0)
            sb.swarmplot(data=dataframe, color="black", size=8)

            # label the y axis
            plt.ylabel(ylab, fontsize=ylab_size)

            # label the x axis
            plt.xticks(list(range(len(xlab))), xlab)

            # determine fontsize of x and y ticks
            plot.tick_params(axis="x", labelsize=28)
            plot.tick_params(axis="y", labelsize=28)

            plt.show()

        self.place_button(ma_root, "Plot", start_analysis, 195, 400, 30, 110)

        ma_root.mainloop()


class CorrelationAnalysis(Control):

    def __init__(self):
        Control.__init__(self)

    def corr_analysis(self):

        ca_root = Tk()

        self.new_window(ca_root, "MitoSegNet Analyser - Correlation analysis", 500, 650)
        self.small_menu(ca_root)

        table_path1 = StringVar(ca_root)
        table1_name = StringVar(ca_root)
        table_path2 = StringVar(ca_root)
        table2_name = StringVar(ca_root)

        stat_value = StringVar(ca_root)

        ss = StringVar(ca_root)
        nb = StringVar(ca_root)
        bl = StringVar(ca_root)
        tbl = StringVar(ca_root)
        ci = StringVar(ca_root)
        ar = StringVar(ca_root)
        min_al = StringVar(ca_root)
        maj_al = StringVar(ca_root)
        ecc = StringVar(ca_root)
        per = StringVar(ca_root)
        sol = StringVar(ca_root)
        mean_int = StringVar(ca_root)
        max_int = StringVar(ca_root)
        min_int = StringVar(ca_root)

        def askopentable1():
            set_tablepath1 = tkinter.filedialog.askopenfilename(parent=ca_root, title='Select file')
            table_path1.set(set_tablepath1)

        def askopentable2():
            set_tablepath2 = tkinter.filedialog.askopenfilename(parent=ca_root, title='Select file')
            table_path2.set(set_tablepath2)

        #### browse for first table location

        self.place_text(ca_root, "Select measurements table file 1", 15, 20, None, None)
        self.place_button(ca_root, "Browse", askopentable1, 435, 50, 30, 50)
        self.place_entry(ca_root, table_path1, 25, 50, 30, 400)

        #### enter table 1 name

        self.place_text(ca_root, "Enter name of table 1 (e. g. wild type)", 15, 90, None, None)
        self.place_entry(ca_root, table1_name, 25, 110, 30, 400)

        #### browse for second table location (to compare against table 1)

        self.place_text(ca_root, "Select measurements table file 2 (to compare against table 1)", 15, 160, None, None)
        self.place_button(ca_root, "Browse", askopentable2, 435, 190, 30, 50)
        self.place_entry(ca_root, table_path2, 25, 190, 30, 400)

        #### enter table 2 name

        self.place_text(ca_root, "Enter name of table 2 (e. g. mutant)", 15, 230, None, None)
        self.place_entry(ca_root, table2_name, 25, 250, 30, 400)

        descriptor_list, stat_list = control_class.stat_and_desc()

        stat_value.set("Average")
        Label(ca_root, text="Select statistical value to display", bd=1).place(bordermode=OUTSIDE, x=15, y=300)
        popupMenu_stat = OptionMenu(ca_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=25, y=320, height=30, width=160)

        def place_checkbutton(window, text, variable, x, y, width):
            variable.set(False)
            hf_button = Checkbutton(window, text=text, variable=variable, onvalue=True, offvalue=False)
            hf_button.place(bordermode=OUTSIDE, x=x, y=y, height=30, width=width)

        place_checkbutton(ca_root, "Save statistics", ss, 300, 320, 150)

        self.place_text(ca_root, "Select up to 4 variables to compare against each other", 15, 370, None, None)

        # left side
        width = 200
        place_checkbutton(ca_root, "Number of branches", nb, 25, 400, 150)
        place_checkbutton(ca_root, "Branch length", bl, 175, 400, 150)
        place_checkbutton(ca_root, "Total branch length", tbl, 300, 400, 180)

        place_checkbutton(ca_root, "Curvature index", ci, 25, 430, 123)
        place_checkbutton(ca_root, "Solidity", sol, 175, 430, 115)
        place_checkbutton(ca_root, "Mean Intensity", mean_int, 300, 430, 155)

        place_checkbutton(ca_root, "Min Intensity", min_int, 25, 460, 110)
        place_checkbutton(ca_root, "Area", ar, 175, 460, 100)
        place_checkbutton(ca_root, "Minor Axis Length", min_al, 325, 460, 120)

        place_checkbutton(ca_root, "Major Axis Length", maj_al, 32, 490, 120)
        place_checkbutton(ca_root, "Eccentricity", ecc, 193, 490, 100)
        place_checkbutton(ca_root, "Perimeter", per, 313, 490, 100)

        place_checkbutton(ca_root, "Max Intensity", max_int, 30, 520, 100)

        def pair_analysis():

            # descriptor_list, stat_list = control_class.stat_and_desc()

            descriptor_list = ["Number of branches", "Branch length", "Total branch length", "Curvature index",
                               "Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter",
                               "Solidity", "Mean Intensity", "Max Intensity", "Min Intensity", "Image", "Data"]

            variable_list = [nb, bl, tbl, ci, ar, min_al, maj_al, ecc, per, sol, mean_int, max_int, min_int]

            def read_reshape_table(table_path, table_name):

                fin_table = pd.DataFrame(columns=descriptor_list)

                table1_ba = pd.read_excel(table_path, sheet_name="BranchAnalysis")
                table1_sd = pd.read_excel(table_path, sheet_name="ShapeDescriptors")

                table_ba = table1_ba[["Image", "Measurement", stat_value.get()]]
                table_sd = table1_sd[["Image", "Measurement", stat_value.get()]]

                img_list = list(set(table_ba["Image"]))

                for count, img in enumerate(img_list):

                    new_table_ba = table_ba[table1_ba["Image"] == img]
                    new_table_sd = table_sd[table1_sd["Image"] == img]

                    new_table_ba = new_table_ba.transpose()
                    new_table_sd = new_table_sd.transpose()

                    new_table = pd.concat([new_table_ba, new_table_sd], axis=1, sort=False)
                    temp_list = new_table.iloc[2].tolist()

                    temp_list.append(img)
                    temp_list.append(table_name)

                    fin_table.loc[count] = temp_list

                return fin_table

            table = read_reshape_table(table_path1.get(), table1_name.get())
            table2 = read_reshape_table(table_path2.get(), table2_name.get())

            final_table = table2.append(table)

            # print(final_table)

            selection_list = ["Data"]
            for name, var in zip(descriptor_list, variable_list):

                if int(var.get()) == 1:
                    selection_list.append(name)

                if len(selection_list) == 5:
                    break

            final_table = final_table[selection_list]
            sb.pairplot(final_table, hue="Data")

            selection_list.remove("Data")

            table_l = []
            x_l = []
            y_l = []
            pval_norm_x_l = []
            pval_norm_y_l = []
            corr_test_l = []
            corr_co_l = []
            corr_pval_l = []

            def check_correlation(table_name):

                for a, b in itertools.combinations(selection_list, 2):

                    x = final_table[final_table["Data"] == table_name][a]
                    y = final_table[final_table["Data"] == table_name][b]

                    p_norm_a = normaltest(x)[1]
                    p_norm_b = normaltest(y)[1]

                    if p_norm_a > 0.05 and p_norm_b > 0.05:

                        corr_co = pearsonr(x, y)[0]
                        corr_pval = pearsonr(x, y)[1]

                        print(table_name, ",", a, "vs", b, "Pearson Results: correlation",  '%.3f' % corr_co, "p-value",
                              '%.5f' % corr_pval)

                        corr_test_l.append("Pearson")

                    else:

                        corr_co = spearmanr(x, y)[0]
                        corr_pval = spearmanr(x, y)[1]

                        print(table_name, ",", a, "vs", b, "Spearman Results: correlation", '%.3f' % corr_co, ", p-value",
                              '%.5f' % corr_pval)

                        corr_test_l.append("Spearman")

                    table_l.append(table_name)
                    x_l.append(a)
                    y_l.append(b)
                    pval_norm_x_l.append(p_norm_a)
                    pval_norm_y_l.append(p_norm_b)
                    corr_co_l.append(corr_co)
                    corr_pval_l.append(corr_pval)

                print("\n")

                return table_l, x_l, y_l, pval_norm_x_l, pval_norm_y_l, corr_test_l, corr_co_l, corr_pval_l

            table_l, x_l, y_l, pval_norm_x_l, pval_norm_y_l, corr_test_l, corr_co_l, corr_pval_l = check_correlation(table1_name.get())
            table_l, x_l, y_l, pval_norm_x_l, pval_norm_y_l, corr_test_l, corr_co_l, corr_pval_l = check_correlation(table2_name.get())

            if int(ss.get()) == 1:

                stat_table = pd.DataFrame(columns=["Table name", "Variable x", "Variable y", "Normality p-value of x",
                                                   "Normality p-value of y", "Correlation test", "Correlation coefficient",
                                                   "Correlation p-value"])

                stat_table["Table name"] = table_l
                stat_table["Variable x"] = x_l
                stat_table["Variable y"] = y_l
                stat_table["Normality p-value of x"] = pval_norm_x_l
                stat_table["Normality p-value of y"] = pval_norm_y_l
                stat_table["Correlation test"] = corr_test_l
                stat_table["Correlation coefficient"] = corr_co_l
                stat_table["Correlation p-value"] = corr_pval_l

                dir_name = os.path.dirname(table_path1.get())

                stat_table.to_excel(dir_name + os.sep + "Correlation_analysis_" + table1_name.get() + "_" +
                                    table2_name.get() + ".xlsx")

            plt.title(stat_value.get())
            plt.show()

        self.place_button(ca_root, "Start analysis", pair_analysis, 195, 580, 30, 110)
        ca_root.mainloop()


class MultiMorphComparison(Control):

    def __init__(self):
        Control.__init__(self)

    def morph_comparison_table(self):

        mma_root = Tk()

        self.new_window(mma_root, "MitoSegNet Analyser - Get statistics of morphological comparison", 500, 270)
        self.small_menu(mma_root)

        dir_path = StringVar(mma_root)
        stat_value = StringVar(mma_root)

        def askopendir():
            set_dirpath = tkinter.filedialog.askdirectory(parent=mma_root, title='Choose a directory')
            dir_path.set(set_dirpath)

        #### browse for table saving location
        self.place_text(mma_root, "Select directory in which all tables are located", 15, 30, None, None)
        self.place_button(mma_root, "Browse", askopendir, 435, 60, 30, 50)
        self.place_entry(mma_root, dir_path, 25, 60, 30, 400)

        # select statistical value
        descriptor_list, stat_list = control_class.stat_and_desc()

        stat_value.set("Average")
        Label(mma_root, text="Select statistical value to analyse", bd=1).place(bordermode=OUTSIDE, x=15, y=120)
        popupMenu_stat = OptionMenu(mma_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=25, y=140, height=30, width=160)

        def multi_sample_table():

            folder_path = dir_path.get()
            stat_val = stat_value.get()

            table_list = os.listdir(folder_path)

            stat_frame = pd.DataFrame()

            parent_path = os.path.dirname(folder_path)
            writer = pd.ExcelWriter(parent_path + os.sep + stat_val + "_Stat_Table.xlsx", engine='xlsxwriter')

            descriptor_list, stat_list = control_class.stat_and_desc()

            data_dic = {}
            for n in descriptor_list:
                data_dic.update({n: pd.DataFrame(columns=table_list)})

            stat_frame["Descriptor"] = descriptor_list

            print("Analysing ...\n")

            for table_name in table_list:

                print(table_name)

                table_ba = pd.read_excel(folder_path + os.sep + table_name, sheet_name="BranchAnalysis")
                table_sd = pd.read_excel(folder_path + os.sep + table_name, sheet_name="ShapeDescriptors")

                max_vals = []

                norm_p = []

                for desc in descriptor_list:

                    if desc == "Number of branches" or desc == "Branch length" or desc == "Total branch length" or desc == "Curvature index":

                        table = table_ba

                    else:
                        table = table_sd

                    # how to acess measurements
                    meas_table = table[table["Measurement"] == desc]

                    # how to acess statistical values
                    values_list = meas_table[stat_val].tolist()

                    if len(values_list) < 8:
                        raise ValueError('Number of samples needs to be 8 or higher for statistical analysis')

                    ######
                    max_vals.append(np.max(values_list))
                    ######

                    norm_p.append(normaltest(values_list)[1])

                    data_dic[desc][table_name] = values_list

                stat_frame[table_name + " normality test p-value"] = norm_p

            stat_frame.to_excel(writer, sheet_name="Statistics Summary")

            hyp_test = []
            hyp_p = []

            for desc in data_dic:

                data_dic[desc].to_excel(writer, sheet_name="sing_vals_" + desc)

                l = []
                normal = []
                for name in data_dic[desc]:
                    l.append(data_dic[desc][name].tolist())

                    p_norm = normaltest(data_dic[desc][name].tolist())[1]

                    if p_norm > 0.05:
                        normal.append(True)
                    else:
                        normal.append(False)

                # tests null hypothesis that all input samples are from populations with equal variance
                p_lev = levene(*l)[1]

                ident = False
                for a, b in itertools.combinations(l, 2):

                    if a == b:
                        ident = True

                # unpacking list using the *args syntax
                """
                *args and **kwargs allow you to pass a variable number of arguments to a function. 
                """

                if ident == False:

                    if False in normal:
                        p_val = kruskal(*l)[1]

                        hyp_test.append("Kruskal-Wallis test")

                    else:

                        # since no welch's ANOVA is implemented in python i am alternatively using the kruskal-wallis test
                        # for unequal variance (but normally distributed)
                        if p_lev > 0.05:
                            p_val = f_oneway(*l)[1]
                            hyp_test.append("one-way ANOVA")

                        else:
                            p_val = kruskal(*l)[1]
                            hyp_test.append("Kruskal-Wallis test")

                else:

                    p_val = 1
                    hyp_test.append("No test possible")

                hyp_p.append(p_val)

            stat_frame["Hypothesis test"] = hyp_test
            stat_frame["Hypothesis test p-value"] = hyp_p

            stat_frame.to_excel(writer, sheet_name="Statistics Summary")

            tkinter.messagebox.showinfo("Done", "Table generated", parent=mma_root)

            writer.save()

        self.place_button(mma_root, "Create table", multi_sample_table, 195, 200, 30, 110)

        mma_root.mainloop()

    def morph_comparison_plot(self):

        mmp_root = Tk()

        self.new_window(mmp_root, "MitoSegNet Analyser - Get plot of morphological comparison", 500, 250)
        self.small_menu(mmp_root)

        dir_path = StringVar(mmp_root)
        stat_value = StringVar(mmp_root)
        descriptor = StringVar(mmp_root)

        def askopendir():
            set_dirpath = tkinter.filedialog.askdirectory(parent=mmp_root, title='Choose a directory')
            dir_path.set(set_dirpath)

        #### browse for table saving location
        self.place_text(mmp_root, "Select directory in which all tables are located", 15, 30, None, None)
        self.place_button(mmp_root, "Browse", askopendir, 435, 60, 30, 50)
        self.place_entry(mmp_root, dir_path, 25, 60, 30, 400)

        descriptor_list, stat_list = control_class.stat_and_desc()

        descriptor.set("Area")
        Label(mmp_root, text="Select shape descriptor to plot", bd=1).place(bordermode=OUTSIDE, x=15, y=100)
        popupMenu_desc = OptionMenu(mmp_root, descriptor, *set(descriptor_list))
        popupMenu_desc.place(bordermode=OUTSIDE, x=25, y=120, height=30, width=160)

        stat_value.set("Average")
        Label(mmp_root, text="Select statistical value to display", bd=1).place(bordermode=OUTSIDE, x=255, y=100)
        popupMenu_stat = OptionMenu(mmp_root, stat_value, *set(stat_list))
        popupMenu_stat.place(bordermode=OUTSIDE, x=265, y=120, height=30, width=160)

        def multi_sample_plot():

            desc = descriptor.get()

            folder_path = dir_path.get()
            stat_val = stat_value.get()

            table_list = os.listdir(folder_path)

            new_table_list = []
            for i in table_list:
                new_table_list.append(i.split("_")[0])

            ylab = stat_val + " " + desc.lower()
            ylab_size = 34
            xlab = new_table_list

            dataframe = pd.DataFrame(columns=table_list)

            # stat_frame["Descriptor"] = descriptor_list

            for table_name in table_list:

                table_ba = pd.read_excel(folder_path + os.sep + table_name, sheet_name="BranchAnalysis")
                table_sd = pd.read_excel(folder_path + os.sep + table_name, sheet_name="ShapeDescriptors")

                max_vals = []
                # norm_p = []

                # for desc in descriptor_list:

                if desc == "Number of branches" or desc == "Branch length" or desc == "Total branch length" or desc == "Curvature index":

                    table = table_ba

                else:
                    table = table_sd

                # how to acess measurements
                meas_table = table[table["Measurement"] == desc]

                # how to acess statistical values
                values_list = meas_table[stat_val].tolist()

                ######
                max_vals.append(np.max(values_list))
                ######

                dataframe[table_name] = values_list

            # select plot
            plot = sb.boxplot(data=dataframe, color="white", fliersize=0)
            sb.swarmplot(data=dataframe, color="black", size=8)

            # label the y axis
            plt.ylabel(ylab, fontsize=ylab_size)

            # label the x axis
            plt.xticks(list(range(len(xlab))), xlab)

            # determine fontsize of x and y ticks
            plot.tick_params(axis="x", labelsize=28)
            plot.tick_params(axis="y", labelsize=28)

            plt.show()

        self.place_button(mmp_root, "Create plot", multi_sample_plot, 195, 200, 30, 110)

        mmp_root.mainloop()


if __name__ == '__main__':
    """
    Main (starting) window
    """

    control_class = Control()
    get_measurements = Get_Measurements()
    mid_window = MidWindow()

    root = Tk()

    control_class.new_window(root, "MitoA Navigator - Start", 300, 190)
    control_class.small_menu(root)

    control_class.place_button(root, "Get measurements", get_measurements.get_measurements_window, 85, 20, 60, 150)
    control_class.place_button(root, "Analyse", mid_window.midwindow, 85, 100, 60, 150)

    root.mainloop()






