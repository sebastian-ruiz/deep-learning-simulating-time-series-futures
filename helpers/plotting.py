import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from plotly.offline import plot
import plotly.graph_objs as go
from scipy.stats import norm
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
# matplotlib.use('TkAgg')
from config import Config

import matplotlib.ticker as ticker
import matplotlib.dates as dates
from matplotlib.lines import Line2D


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat

        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % ticker._mathdefault(self.format)


class Plotting:

    def __init__(self):
        self.config = Config()

    def plot_2d(self, curves, file_name, curve2=None, time=None, show=True, save=True, title=False, timeseries=False, evaluation=None):

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.grid(True)

        if timeseries:

            print("curves.shape", curves.shape, curves.shape[1])
            NUM_COLORS = curves.shape[1]
            cm = plt.get_cmap('coolwarm')
            ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)[::-1]])
            if time is None:
                ax.plot(curves)
            else:
                ax.plot(time, curves)
            plt.xticks(rotation=20)
            custom_lines = [Line2D([0], [0], color=cm(1.), lw=4),
                            Line2D([0], [0], color=cm(0.), lw=4)]
            ax.legend(custom_lines, ['Short End', 'Long End'])

            ax.set_ylabel("Price ($\$$)")

        else:
            if isinstance(curves, pd.DataFrame):
                ax.plot(curves)
                plt.xticks(rotation=20)
                ax.set_xlabel('Date')
            elif time is None:
                ax.plot(curves)
                ax.set_xlabel('Time (days)')
            else:
                ax.plot(time, curves)
                plt.xticks(rotation=20)
                ax.set_xlabel('Date')

            if curve2 is not None:
                ax.plot(curve2, color='tab:orange')

            ax.set_ylabel('Latent Value')


        if evaluation is not None:

            # if isinstance(evaluation, pd.DataFrame):
            #     _evaluation = evaluation
            # else:
            #     _evaluation = pd.DataFrame(evaluation)

            ax2 = ax.twinx()
            ax2.plot(curves.index.values, evaluation, 'k', label='eval', alpha=0.5)
            ax2.set_ylabel("SMAPE")


        if title:
            plt.title(file_name)
        if save:
            plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
            plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()

    def plot_unrealisticness(self, curves, file_name, time=None, show=True, save=True, title=False, timeseries=False, evaluation=None, eval_label='MSE'):

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.grid(True)

        if timeseries:

            print("curves.shape", curves.shape, curves.shape[1])
            NUM_COLORS = curves.shape[1]
            cm = plt.get_cmap('coolwarm')
            ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)[::-1]])
            if time is None:
                ax.plot(curves,linestyle='None',marker='o')
            else:
                ax.plot(time, curves,linestyle='None',marker='o')
            plt.xticks(rotation=20)
            custom_lines = [Line2D([0], [0], color=cm(1.), lw=4),
                            Line2D([0], [0], color=cm(0.), lw=4)]

            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 - box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(custom_lines, ['Short End', 'Long End'], loc='upper center', bbox_to_anchor=(0.5, 1.15),
                      fancybox=True, shadow=False, ncol=5)
            fig.subplots_adjust(bottom=0.2)

            # ax.legend(custom_lines, ['Short End', 'Long End'])

        else:
            if time is None:
                ax.plot(curves,linestyle='None',marker='o')
                ax.set_xlabel('Time (days)')
            else:
                ax.plot(time, curves,linestyle='None',marker='o')
                plt.xticks(rotation=20)
                ax.set_xlabel('Date')

            ax.set_ylabel('Latent Value')

        if evaluation is not None:
            ax2 = ax.twinx()
            ax2.plot(evaluation, 'k', label='eval', alpha=1)
            # ax2.bar(np.arange(len(evaluation)), evaluation, color='k', alpha=0.5)
            ax.set_ylabel("Price ($\$$)")
            ax2.set_ylabel(eval_label)

            box2 = ax2.get_position()
            ax2.set_position([box2.x0, box2.y0 - box2.height * 0.1,
                             box2.width, box2.height * 0.9])

        if title:
            plt.title(file_name)
        if save:
            plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
            plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()


    def plot_some_curves(self, file_name, surface, output=None, curves=None, maturities=None, plot_separate=False):

        nrow = 2
        ncol = 2
        if maturities is None:
            _maturities = np.arange(surface.shape[1])
        else:
            _maturities = maturities
        if curves is None:
            _curves = [0, 300, 600, 700]
        else:
            _curves = curves

        if isinstance(surface, pd.DataFrame):
            _surface = surface
        else:
            _surface = pd.DataFrame(surface)

        if isinstance(output, pd.DataFrame):
            _output = output
        else:
            _output = pd.DataFrame(output)

        len_curves = len(curves)

        if len_curves % 2 != 0:
            return
        else:
            ncol = 4

        if plot_separate:
            nrow = int(len_curves / 2)
            _curves = np.append(_curves, _curves)
        else:
            nrow = int(len_curves / 4)

        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2 * ncol, 2 * nrow), sharex=True) # figsize: width, height

        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False, grid_alpha=0.0)

        for (i, ax) in zip(np.arange(len(_curves)), axs.reshape(-1)):
            if i % ncol == 0:
                ax.set_ylabel("Price ($\$$)", labelpad=30)
            # if i >= len_curves - ncol:
                # ax.set_xlabel("Maturities (years)")
            ax.grid(True)
            if plot_separate:
                if i < len_curves:
                    line_1, = ax.plot(_maturities, _surface.iloc[_curves[i]], color='tab:blue')
                    ax.set_title(_surface.index[_curves[i]])
                else:
                    if i % ncol == 0:
                        ax.set_ylabel("Scaled", labelpad=30)
                    line_2, = ax.plot(_maturities, _output.iloc[_curves[i]], color='tab:orange')
                    # ax.set_title(_output.index[_curves[i]])
            else:
                line_1, = ax.plot(_maturities, _surface.iloc[_curves[i]], color='tab:blue')
                if output is not None:
                    line_2, = ax.plot(_maturities, _output.iloc[_curves[i]], color='tab:orange')
                ax.set_title(_surface.index[_curves[i]])

            ax.set_xticks(np.arange(min(_maturities), max(_maturities)+1, 2.0))

        plt.xlabel("Term to maturity (years)", labelpad=30)
        plt.tight_layout()
        # plt.title(file_name)
        plt.savefig(self.config.get_filepath_img(file_name), dpi=300, transparent=True)
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()



    def plot_loss(self, loss, val_loss, file_name):

        plt.semilogy(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title(file_name)
        plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()

    def plot_grid_1dim(self, maturities, predict, file_name, show=True):
        # # display a 2D manifold of the digits
        num_rows = 5
        num_columns = 6
        n = num_rows * num_columns  # figure with 15x15 digits
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))

        plt.figure(figsize=(num_rows * 2, num_columns * 2), dpi=100)

        for i, xi in enumerate(grid_x):
            z_sample = np.array([[xi]])
            x_decoded = predict(z_sample)
            ax = plt.subplot(num_rows, num_columns, i + 1)
            if i == 0:
                ax.set_yticklabels([])

            ax.set_xticklabels([])

            plt.plot(maturities, x_decoded[0])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.title(file_name)
        plt.savefig(self.config.get_filepath_img(file_name), dpi=100, bbox_inches='tight')
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()

    def plot_grid_2dim(self, maturities, decoder, file_name, preprocess=None, dataset_name=None, n=8, show=True):
        # # display a 2D manifold of the digits
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        linspace = np.linspace(0.05, 0.95, n)
        grid_x = norm.ppf(linspace)
        grid_y = norm.ppf(linspace)

        fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(n, n), sharex=True)
        fig.add_subplot(111, frameon=False)
        # plt.xticks(np.arange(min(linspace), max(linspace) + 0.1, 0.15))
        # plt.yticks(np.arange(min(linspace), max(linspace) + 0.1, 0.15))
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False, grid_alpha=0.0)
        for i, yi in enumerate(np.flip(grid_y)):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                if preprocess is not None and dataset_name is not None:
                    x_decoded = preprocess.rescale_data(x_decoded, dataset_name=dataset_name)

                # axs[i, j].set_xticks(np.arange(min(maturities), max(maturities) + 1, 2.0))
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticks([], [])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticks([], [])

                if j <= 0:
                    axs[i, j].set_ylabel(str(round(yi, 2)), rotation=0, labelpad=15)
                if i >= n - 1:
                    axs[i, j].set_xlabel(str(round(xi, 2)))

                axs[i, j].plot(maturities, x_decoded[0])

        # if preprocess is not None and dataset_name is not None:
        #     plt.ylabel("Price ($\$$)", labelpad=30)
        # else:
        #     plt.ylabel("Scaled Price", labelpad=30)
        #
        # plt.xlabel("Term to maturity (years)", labelpad=30)

        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(self.config.get_filepath_img(file_name), dpi=100)  # , bbox_inches='tight'
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def format_date(x, pos=None):
        return dates.num2date(x).strftime('%d-%m-%Y')  # use FuncFormatter to format dates

    def plot_3d(self, file_name, surface, maturities=None, time=None, x_axis=None, y_axis=None, z_axis=None, show_title=False):

        if isinstance(surface, pd.DataFrame):
            # print("surface.columns.values", surface.columns.values)
            # print("surface.index.values", surface.index.values)

            _maturities = surface.columns.values/(30 * 12) # for years
            _time = matplotlib.dates.date2num(surface.index.values)
        else:
            if maturities is None:
                _maturities = np.arange(surface.shape[1])
            else:
                _maturities = maturities
            if time is None:
                _time = np.arange(surface.shape[0])
            else:
                _time = matplotlib.dates.date2num(time)

        if x_axis is None:
            x_axis = "Term to maturity (years)"
        if y_axis is None:
            if time is not None or isinstance(surface, pd.DataFrame):
                y_axis = "Time"
            else:
                y_axis = "Time (days)"
        if z_axis is None:
            z_axis = "Price ($\$$)"

        # 3d plot
        xx, yy = np.meshgrid(_maturities, _time)
        fig = plt.figure()
        ax = Axes3D(fig, rect=[-0.05, 0.03, 1, 1])  # make room for date labels

        ax.plot_surface(xx, yy, surface, rstride=1, cstride=1, cmap=cm.jet,
                        linewidth=0.1)  # cstride=1, rstride=1, # cstride=1, rstride=5,
        ax.set_xlabel(x_axis)
        # y_labels_at_steps = 80
        if time is not None or isinstance(surface, pd.DataFrame):
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_date))

            for tl in ax.yaxis.get_ticklabels():  # re-create what autofmt_xdate but with w_xaxis
                tl.set_ha('left')
                # tl.set_rotation(-10)

            ax.set_ylabel(y_axis, labelpad=40)

        else:
            ax.set_ylabel(y_axis)


        ax.set_zlabel(z_axis)
        if show_title:
            plt.title(file_name)
        fig.savefig(self.config.get_filepath_img(file_name), dpi=300, transparent=True) # , bbox_inches='tight'
        fig.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()

    def plotly_3d(self, file_name, surface, surface_conditioned=None, maturities=None, time=None):

        if surface_conditioned is not None:
            temp = np.concatenate((surface_conditioned, np.zeros(surface.shape[1])[np.newaxis, :]))
            surface = np.concatenate((temp, surface))
        if maturities is None:
            maturities = np.arange(surface.shape[1])
        if time is None:
            time = np.arange(surface.shape[0])
        data = [
            go.Surface(x=maturities, y=time, z=surface, colorscale='Jet')
        ]
        layout = go.Layout(
            # title='WTI Nymex',
            autosize=True,
            scene=dict(
                xaxis=dict(
                    title='maturity'),
                yaxis=dict(
                    title='t'),
                zaxis=dict(
                    title='price'),
            ),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=self.config.get_filepath('/assets/images', file_name + '.html'), auto_open=False)

    def plot_3d_comparison_simple(self, maturities, input, decoded, num_samples, file_name):

        index = np.arange(len(input))
        x_test_shuffled, decoded_data_shuffled, index_shuffled = shuffle(input, decoded, index)

        self.plot_3d_comparison(maturities, np.arange(input.shape[1]), x_test_shuffled[: num_samples, :],
                                    decoded_data_shuffled[: num_samples, :], num_samples, file_name,
                                    index_shuffled)

    def plot_3d_comparison(self, maturities, time, surfaces_input, surfaces_output, num_samples, file_name, titles=None):
        xx, yy = np.meshgrid(maturities, time)

        n = num_samples  # how many digits we will display
        fig = plt.figure(figsize=(int(n/2)*2, 2*2), dpi=100)
        for i in range(n):
            ax = fig.add_subplot(2, n, i + 1, projection='3d')
            ax.plot_surface(xx, yy, surfaces_input[i], rstride=5, cstride=1, cmap=cm.jet,
                            linewidth=0.1)  # cstride=1, rstride=1,

            if titles is not None:
                ax.set_title("input: " + str(titles[i]))

            ax = fig.add_subplot(2, n, i + n + 1, projection='3d')
            ax.plot_surface(xx, yy, surfaces_output[i], rstride=5, cstride=1, cmap=cm.jet,
                            linewidth=0.1)  # cstride=1, rstride=1,

            if titles is not None:
                ax.set_title("output: " + str(titles[i]))
            # display original
            # plt.subplot(2, int(n/2), i + 1)
            # line_1, = plt.plot(maturities, surfaces_input[i], label="$X_{i}$")
            # line_2, = plt.plot(maturities, surfaces_output[i], label="$\hat X_{i}$")
            # plt.plot_surface

        # fig.legend((line_1, line_2), ("$X_{i}$", "$\hat X_{i}$"), "upper right")
        plt.title(file_name)
        plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()

    def plot_3d_many(self, file_name, data, maturities=None, time=None, show=True, save=True):

        if maturities is None:
            maturities = np.arange(data[0].shape[1])
        if time is None:
            time = np.arange(data[0].shape[0])

        xx, yy = np.meshgrid(maturities, time)

        n = data.shape[0]  # how many digits we will display
        nrow = 2
        ncol = int(n/2) # rounds down
        fig, axs = plt.subplots(nrow, ncol)
        for i, ax in enumerate(fig.axes):
            ax.plot_surface(xx, yy, data[i], rstride=5, cstride=1, cmap=cm.jet, linewidth=0.1)  # cstride=1, rstride=1,
            ax.set_ylabel(str(i))

        plt.title(file_name)
        if save:
            plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
            plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()

    def plot_3d_cov(self, file_name, surface, maturities=None, x_axis=None, y_axis=None, z_axis=None,
                show_title=False):

        if maturities is None:
            maturities = np.arange(surface.shape[1])/12  # /12 to get it in years

        if x_axis is None:
            # x_axis = "Tenors"
            x_axis = "Term to maturity (years)"
        if y_axis is None:
            # y_axis = "Tenors"
            y_axis = "Term to maturity (years)"
        if z_axis is None:
            z_axis = "Covariance"

        # 3d plot
        xx, yy = np.meshgrid(maturities, maturities)
        fig = plt.figure()
        ax = Axes3D(fig)  # , rect=[-0.05, 0.03, 1, 1] make room for date labels

        ax.plot_surface(xx, yy, surface, rstride=1, cstride=1, cmap=cm.jet,
                                linewidth=0.1)
        # ax.plot_wireframe(xx, yy, surface, rstride=3, cstride=3)  # cstride=1, rstride=1,
        ax.set_xlabel(x_axis)
        ax.zaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        if show_title:
            plt.title(file_name)
        fig.savefig(self.config.get_filepath_img(file_name), dpi=300)  # , bbox_inches='tight'
        fig.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()

    def plot_losses(self, discriminator_loss, generator_loss, generator_mse, file_name, legend=None):
        if legend is None:
            legend = ['discriminator loss', 'generator loss', 'generator mse']
        plt.semilogy(discriminator_loss)
        plt.semilogy(generator_loss)
        plt.semilogy(generator_mse)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(legend, loc='upper left')
        plt.title(file_name)
        plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()

    def plot_training_sample(self, file_name, generated, real, len_condition, after_real_data=False, show=True):
        'expects first dimension to be batch size'

        # print('real', real.index.values)
        dates = matplotlib.dates.date2num(real.index.values)

        fig, ax = plt.subplots()

        x_values = np.arange(len(generated[0]))
        if after_real_data:
            x_values = dates[len_condition:len(generated[0]) + len_condition]
            # x_values = np.arange(len_condition, len(generated[0]) + len_condition)
            # print('x_values ', x_values, len(real[0]), len(generated[0]))

        if generated.shape[0] > 1:
            plt.plot(x_values, generated[1:].transpose(), c='k', alpha=.1)
            plt.plot(x_values, generated[0], c='b')
        else:
            plt.plot(x_values, generated[0], c='b')

        plt.plot(dates, np.array(real), c='r')

        plt.xlabel("Time")
        plt.ylabel("Price ($\$$)")

        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=15))

        plt.grid(True)
        plt.xticks(rotation=20)

        plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()

    def plot_3d_training(self, file_name, generated, real, show=True, after_real_data=False):
        if len(generated.shape) is 3:
            for i in np.arange(generated.shape[0]):
                self._plot_3d_training(file_name, generated[i], real, show=True, after_real_data=False)
        else:
            self._plot_3d_training(file_name, generated, real, show=True, after_real_data=False)

    def _plot_3d_training(self, file_name, generated, real, show=True, after_real_data=False):

        # 3d plot
        xx, yy = np.meshgrid(np.arange(generated.shape[1]), np.arange(generated.shape[0]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, generated, cmap=cm.jet, cstride=1, rstride=1)  # ,linewidth=0.1 ,antialiased=False
        ax.set_xlabel('Maturity (days)')
        y_labels_at_steps = 80
        # ax.set_yticks(y_labels[::y_labels_at_steps])
        # ax.set_yticklabels(np.array(df["date"].values)[::y_labels_at_steps])
        ax.set_ylabel('Time (days)')
        ax.set_zlabel('Price ($\$$)')
        # plt.title(file_name)
        fig.savefig(self.config.get_filepath_img(file_name), dpi=300, transparent=True)
        fig.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        if show:
            plt.show()
        plt.close()



    def plot_data(self, harvest_data, file_name):

        df = pd.DataFrame.from_dict(harvest_data, orient='index')
        df.reset_index(inplace=True)  # make date a column
        df.drop(columns=[1], inplace=True)  # drop duplicate column
        df.rename(index=str, columns={"index": "date", 0: "curve"}, inplace=True)
        df.sort_values(by='date', inplace=True)  # sort by date

        first_date = df["date"].iloc[0]
        first_curve = df["curve"].iloc[0]
        end_date = df["date"].iloc[-1]

        # compute maximum date to plot to
        # max_end_date = np.maximum(int(first_curve.x[-1]), convert_dates_to_days(end_date, start_date=first_date))
        # print("max end date max of: ", int(first_curve.x[-1]), convert_dates_to_days(end_date, start_date=first_date))
        # take first_date as our starting point
        # first_curve.x[-1] is the last x value point on the first curve

        surface = []
        end_of_x_range = []
        for index, row in df.iterrows():
            curve = row["curve"]
            # print(curve)
            end_of_x_range.append(curve.x[-1])
            # print(curve.x[-1])

        # for br_ice this is around 1800
        max_x_values = np.amax(np.asarray(end_of_x_range))

        # we choose 1680 because then we get 56 = 2^3 * 7 features. We want it to be divisible by two a lot to use CNN.
        x_labels = np.arange(start=0, stop=1680, step=30)
        # y_labels = pltdates.date2num(np.array(df["date"].values))
        # y_labels = np.array(convert_dates_to_days(np.array(df["date"].values), start_date=first_date))

        print("max x values:", max_x_values)

        for index, row in df.iterrows():
            curve = row["curve"]
            interp_values = curve(x_labels)
            surface.append(interp_values)

        surface = np.array(surface)

        print("surface.shape", surface.shape)
        print("x_labels.shape", x_labels.shape)

        df_surface = pd.DataFrame(surface, index=np.array(df["date"].values), columns=x_labels)
        df_surface.to_pickle("data.pkl")

        # print(df_surface)

        # print(surface)
        # print(df["date"].values)

        nrow = 2
        ncol = 2
        plot_curves = [0, 300, 600, 700]
        plot_names = []
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol)

        for (i, ax) in zip(np.arange(len(plot_curves)), axs.reshape(-1)):
            if i == 0 or i == 2:
                ax.set_ylabel("price dollar/$")
            if i == 2 or i == 3:
                ax.set_xlabel("maturities")
            ax.grid(True)
            ax.set_title(df["date"].values[plot_curves[i]])
            print(df["date"].values[plot_curves[i]])
            ax.plot(x_labels, surface[plot_curves[i]])

        plt.tight_layout()
        plt.title(file_name)
        plt.savefig(self.config.get_filepath_img(file_name), bbox_inches='tight', dpi=300)
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)  # , bbox_inches='tight'
        plt.show()
        plt.close()


        # self.plot_3d_comparison(x_labels, np.array(df["date"].values), surfaces_input, surfaces_output, num_samples, file_name, titles=None)

        # xx, yy = np.meshgrid(x_labels, y_labels)
        # xx = xx.transpose()
        # yy = yy.transpose()


        # xx, yy = np.mgrid(x_labels, y_labels)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(xx, yy, surface, rstride=5, cstride=1, cmap=cm.jet, linewidth=0.1) #cstride=1, rstride=1,
        # ax.set_xlabel('maturity')
        # y_labels_at_steps = 80
        # ax.set_yticks(y_labels[::y_labels_at_steps])
        # ax.set_yticklabels(np.array(df["date"].values)[::y_labels_at_steps])
        # ax.set_ylabel('t')
        # ax.set_zlabel('price')
        # fig.savefig('plot_harvester.png', bbox_inches='tight', dpi=300)
        # plt.show()

        self.plotly_3d("plot_harvester", surface, x_labels, np.array(df["date"].values))
        self.plot_3d("plot_harvester", surface, x_labels, np.array(df["date"].values))

    def plot_space(self, maturities, vae, file_name, latent_dim=2):

        if latent_dim == 1:
            # display a 2D manifold of the digits
            n = 20  # figure with 15x15 digits
            # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
            # to produce values of the latent variables z, since the prior of the latent space is Gaussian
            grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
            # grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

            plt.figure(figsize=(n, 1), dpi=100)

            for i, yi in enumerate(grid_x):
                z_sample = np.array([[yi]])
                x_decoded = vae.decode(z_sample)
                ax = plt.subplot(1, n, i + 1)
                if not i == 0:
                    ax.set_yticklabels([])
                ax.set_xticklabels([])

                plt.plot(maturities, x_decoded[0])

        elif latent_dim == 2:
            # display a 2D manifold of the digits
            n = 20  # figure with 15x15 digits
            # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
            # to produce values of the latent variables z, since the prior of the latent space is Gaussian
            grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

            plt.figure(figsize=(n, n), dpi=100)

            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    z_sample = np.array([[xi, yi]])
                    x_decoded = vae.decode(z_sample)
                    ax = plt.subplot(n, n, j + 1 + i * n)
                    if not j == 0:
                        ax.set_yticklabels([])
                    if not i == n - 1:
                        ax.set_xticklabels([])

                    plt.plot(maturities, x_decoded[0])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(self.config.get_filepath_img(file_name), dpi=300, transparent=True)
        plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)
        plt.show()
