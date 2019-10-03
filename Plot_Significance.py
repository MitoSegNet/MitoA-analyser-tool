import seaborn as sb
import matplotlib.pyplot as plt

# pos_y and pos_x determine position of bar, p sets the number of asterisks, y_dist sets y distance of the asterisk to
# bar, and distance sets the distance between two or more asterisks
#significance_bar(pos_y=1, pos_x=[0, 4], bar_y=0.03, p=1, y_dist=0.02, distance=0.1)

def significance_bar(pos_y, pos_x, bar_y, p, y_dist, distance):

    if p != 0:

        Pos_Line_x = pos_x
        Pos_Line_y = [pos_y, pos_y]

        plt.plot(Pos_Line_x, Pos_Line_y, linewidth=1, color='black')

        plt.plot([pos_x[0], pos_x[0]], [pos_y,pos_y- bar_y], linewidth=1, color='black')
        plt.plot([pos_x[-1], pos_x[-1]], [pos_y, pos_y - bar_y], linewidth=1, color='black')

        n = 0
        for i in range(p):

            if p == 2:

                plt.scatter(x=(pos_x[0]+max(pos_x)) / 2+n-(distance/2), y=pos_y + y_dist, marker="*", color="black")

                n+=distance

            else:

                plt.scatter(x=(pos_x[0]+max(pos_x)) / 2+n, y=pos_y + y_dist, marker="*", color="black")

                if i ==1:
                    n-=(distance*2)

                else:
                    n+=distance


