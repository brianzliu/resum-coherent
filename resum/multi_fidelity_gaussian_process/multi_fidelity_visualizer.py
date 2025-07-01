import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from resum.utilities import plotting_utils as plotting

class MultiFidelityVisualizer():
    def __init__(self, mf_model, parameters, x_fixed):
        self.mf_model = mf_model
        self.parameters = parameters
        self.x_fixed = x_fixed
        self.colors_std = ['darkturquoise','cadetblue','coral']
        self.colors_mean = ['lightseagreen','teal','orangered']

    # Drawings of the model predictions projecting each dimension on a fixed point in space for the remaining dimensions
    def draw_model_projections(self, fig, y_lim=None):
        SPLIT = 100
        ncol=2

        #fig,ax = plt.subplots(nrow,ncol,figsize=(15, 5),constrained_layout=True)
        ax = fig.axes
        
        indices = [i for i in range(len(ax))]
        indices[0], indices[ncol-1] = indices[ncol-1], indices[0]

        for i,p in enumerate(self.parameters):   
            ## Compute mean and variance predictions
            x_plot=[self.x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])

            for f in range(self.mf_model.nfidelities):
                f_mean_mf_model, f_var_mf_model = self.mf_model.model.predict(X_plot[f*SPLIT:(f+1)*SPLIT])
                f_std_mf_model = np.sqrt(f_var_mf_model)

                # Determine label based on fidelity: the first two fidelities (0,1) are CNP, the last one (2) is raw
                if f < 2:
                    label_mean = r'$y_{CNP}$' if (i == 0) else "_nolegend_"  # show only once in legend
                    label_std  = r'$y_{CNP}\;\pm\;1\sigma$' if (i == 0) else "_nolegend_"
                else:
                    label_mean = r'$y_{raw}$' if (i == 0) else "_nolegend_"
                    label_std  = r'$y_{raw}\;\pm\;1\sigma$' if (i == 0) else "_nolegend_"

                ax[indices[i]].fill_between(x_tmp.flatten(), (f_mean_mf_model - f_std_mf_model).flatten(), 
                            (f_mean_mf_model + f_std_mf_model).flatten(), color=self.colors_std[f], alpha=0.1, label=label_std)
                ax[indices[i]].plot(x_tmp,f_mean_mf_model, '--', color=self.colors_mean[f], label=label_mean)

            ax[indices[i]].set_xlabel(p, fontsize=10)
            ax[indices[i]].set_ylabel(r'$y$')
            ax[indices[i]].set_xlim(self.parameters[p][0], self.parameters[p][1])
            
            # Apply optional y-axis limit if provided for zoomed view
            if y_lim is not None:
                ax[indices[i]].set_ylim(*y_lim)

            # After potential limit adjustments, set y-ticks every 0.01
            ymin, ymax = ax[indices[i]].get_ylim()
            ax[indices[i]].set_yticks(np.arange(np.floor(ymin * 100) / 100,
                                               np.ceil(ymax * 100) / 100 + 0.01, 0.01))

        for i in range(len(self.parameters),len(ax)): 
            ax[i].set_axis_off()

        # Add a single shared legend outside of the plotting loop (only once for the whole figure)
        if len(fig.axes) > 0:
            handles, labels = ax[indices[0]].get_legend_handles_labels()
            # Remove duplicate/no-legend entries
            filtered = [(h,l) for h,l in zip(handles,labels) if l != "_nolegend_"]
            if filtered:
                fig.legend([h for h,l in filtered], [l for h,l in filtered], loc="upper center", ncol=len(filtered), frameon=False)

        return fig

    # Drawings of the aquisition function
    def draw_acquisition_func(self, fig, us_acquisition, x_next=np.array([])):
        SPLIT = 50
        ax2 = fig.axes

        for i, p in enumerate(self.parameters):
            ax2[i].set_title(f"Projected acquisition function - {p}")
            x_plot = [self.x_fixed[:] for _ in range(SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(SPLIT):
                x_plot[k][i] = x_tmp[k]
            x_plot = np.atleast_2d(x_plot)
            X_plot = convert_x_list_to_array([x_plot, x_plot])
            
            acq = us_acquisition.evaluate(X_plot[SPLIT:])
            try:
                color = next(ax2[i].get_prop_cycle())["color"]
            except AttributeError:
                color = "blue"  # Fallback color if cycle is unavailable
            
            ax2[i].plot(x_tmp, acq / acq.max(), color=color)
            
            acq = us_acquisition.evaluate(X_plot[:SPLIT])
            ax2[i].plot(x_tmp, acq / acq.max(), color=color, linestyle="--")
            
            if x_next.any():
                ax2[i].axvline(x_next[0, i], color="red", label="x_next", linestyle="--")
                ax2[i].text(
                    x_next[0, i] + 0.5, 0.95,
                    f"x = {round(x_next[0, i], 1)}",
                    color="red", fontsize=8
                )
            
            ax2[i].set_xlabel(p)
            ax2[i].set_ylabel(r"$\mathcal{I}(x)$")

        return fig

    def model_validation(self, x_test, y_test):
            nrows = len(x_test)
            ncols = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 3 * nrows), squeeze=False)

            for f in range(len(x_test)):
                ax = axes[f][0]
                x_test_tmp, y_test_tmp = (np.atleast_2d(x_test[f]), np.atleast_2d(y_test[f]).T)

                counter_1sigma = 0
                counter_2sigma = 0
                counter_3sigma = 0

                mfsm_model_mean = np.empty(shape=[0, 0])
                mfsm_model_std = np.empty(shape=[0, 0])
                y_data=[]
                x=[]
                for i in range(len(x_test_tmp)):

                        SPLIT = 1
                        x_plot = []
                        for j in range(self.mf_model.nfidelities):
                            x_plot.append((np.atleast_2d(x_test_tmp[i])))
                        X_plot = convert_x_list_to_array(x_plot)

                        mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[f*SPLIT:(f+1)*SPLIT])
                        std_mf_model = np.sqrt(var_mf_model)

                        y_data.append(y_test_tmp[i])
                        x.append(i)
                        mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                        mfsm_model_std=np.append(mfsm_model_std,std_mf_model[0,0])
                        
                        if (y_test_tmp[i] < mean_mf_model[0][0]+std_mf_model[0][0]) and (y_test_tmp[i] > mean_mf_model[0][0]-std_mf_model[0][0]):
                                counter_1sigma += 1
                        if (y_test_tmp[i] < mean_mf_model[0][0]+2*std_mf_model[0][0]) and (y_test_tmp[i] > mean_mf_model[0][0]-2*std_mf_model[0][0]):
                                counter_2sigma += 1
                        if (y_test_tmp[i] < mean_mf_model[0][0]+3*std_mf_model[0][0]) and (y_test_tmp[i] > mean_mf_model[0][0]-3*std_mf_model[0][0]):
                                counter_3sigma += 1

                #plt.bar(x=np.arange(len(mfsm_model_mean)), height=mfsm_model_mean, color="lightgray", label='RESuM')
                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-3*mfsm_model_std, y2=mfsm_model_mean+3*mfsm_model_std, color="coral",alpha=0.2, label=r'$\pm 3\sigma$')
                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-2*mfsm_model_std, y2=mfsm_model_mean+2*mfsm_model_std, color="yellow",alpha=0.2, label=r'$\pm 2\sigma$')
                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-mfsm_model_std, y2=mfsm_model_mean+mfsm_model_std, color="green",alpha=0.2, label=r'RESuM $\pm 1\sigma$')
                
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin,ymax*1.05)
                ax.set_ylabel(r'$y_{raw}$')
                ax.plot(x[:],y_data[:],'.',color="black", label="Validation Data")
                mse = mean_squared_error(y_data, mfsm_model_mean)
                text = f"MSE: {mse:.5f} $\pm1\sigma$: {counter_1sigma/len(y_data)*100.:.0f}%  $\pm3\sigma$: {counter_2sigma/len(y_data)*100.:.0f}%  $\pm3\sigma$: {counter_3sigma/len(y_data)*100.:.0f}%"
                plotting.place_text_corner(ax, text, fontsize=11, bbox=dict(edgecolor='gray', facecolor='none', linewidth=0.5))
            
            ax.set_xlabel('Simulation Trial Number')
            legend_elements = [
                Line2D([0], [0], marker='.', color='black', linestyle='None', label='Data'),
                Line2D([0], [0], marker='.', color='white', linestyle='None', label='Model prediction'),
                mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
                mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
                mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')
            ]

        
            fig.legend(handles=legend_elements, loc="upper center", ncol=len(legend_elements), fontsize='medium', frameon=False)
            plt.tight_layout()
            plt.show()
            return fig, [counter_1sigma/len(y_data)*100.,counter_2sigma/len(y_data)*100.,counter_3sigma/len(y_data)*100.,mse]
        
    def draw_model(self):
            nrows = self.mf_model.nfidelities
            ncols = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 5 * nrows), squeeze=False)

            for j,f in enumerate(self.mf_model.trainings_data):
                ax = axes[j][0]
                nsamples = 0
                mfsm_model_mean = np.empty(shape=[0, 0])
                mfsm_model_std = np.empty(shape=[0, 0])
                x_train = self.mf_model.trainings_data[f][0]
                y_train = self.mf_model.trainings_data[f][1]
                x_train, y_train = (np.atleast_2d(x_train), np.atleast_2d(y_train).T)
                y_data=[]
                x=[]
                for i in range(len(x_train)):
                        nsamples += 1
                        SPLIT = 1
                        
                        x_plot = (np.atleast_2d(x_train[i]))
                        X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])
                        
                    
                        mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[j*SPLIT:(j+1)*SPLIT])
                        std_mf_model = np.sqrt(var_mf_model)
                        y_data.append(y_train[i])
                        x.append(i)
                        mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                        mfsm_model_std=np.append(mfsm_model_std,std_mf_model[0,0])

                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-mfsm_model_std, y2=mfsm_model_mean+mfsm_model_std, color=self.colors_std[j],alpha=0.2, label=f'{f} model mean $\pm 1\sigma$')
                ax.plot(np.arange(len(mfsm_model_mean)),mfsm_model_mean,color=self.colors_mean[j], label=f'{f} model mean')
                ax.plot(x[:],y_data[:],'.',color="black", label="Data")
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin,ymax*1.1)
                handles, labels = ax.get_legend_handles_labels()
                
                order = [2,1,0]
                ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9, bbox_to_anchor=(0.665,1.),ncol=3)
                ax.set_ylabel('Data and Model Prediction')
                if j == (self.mf_model.nfidelities-1):
                    ax.set_xlabel('Data Point')
            return fig
    
    def draw_model_marginalized(self, keep_axis=0, grid_steps=10):
            x_grid_list = []
            for p in self.parameters:
                arr = np.linspace(self.parameters[p][0],self.parameters[p][1], grid_steps)
                x_grid_list.append(arr)

            mesh = np.meshgrid(*x_grid_list, indexing='ij')
            points = np.stack([m.flatten() for m in mesh], axis=1)
            mesh_grid_list = points.tolist()

            nrows = self.mf_model.nfidelities
            ncols = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 5 * nrows), squeeze=False)

            for j in range(self.mf_model.nfidelities):
                ax = axes[j][0]
                mfsm_model_mean = []
                mfsm_model_var = []
                x_train = mesh_grid_list
                if j == 0:
                     print(f"Warning: There are {self.mf_model.nfidelities}x {len(x_train)} grid points to process... This can take a while")
                #y_train = self.mf_model.trainings_data[f][1]
                x_train = np.atleast_2d(x_train)

                for i in range(len(x_train)):
                        SPLIT = 1
                        x_plot = (np.atleast_2d(x_train[i]))
                        X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])
                        mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[j*SPLIT:(j+1)*SPLIT])
                        mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                        mfsm_model_var.append(var_mf_model[0, 0])

                y_mean = np.array(mfsm_model_mean)
                y_var = np.array(mfsm_model_var)

                grid_shape = [grid_steps] * len(self.parameters)
                y_mean_grid = y_mean.reshape(grid_shape)
                y_var_grid = y_var.reshape(grid_shape)
                all_grid_axes = list(range(len(self.parameters)))
                marginalize_axes = tuple(ax for ax in all_grid_axes if ax != keep_axis)
                # Compute the marginal mean by averaging over the inactive dimensions.
                y_marginalized = np.mean(y_mean_grid, axis=marginalize_axes)
                # Compute the marginal variance via the law of total variance:
                #   var_marg = mean(var_grid) + variance(y_grid)
                y_var_marginalized = np.mean(y_var_grid, axis=marginalize_axes) + np.var(y_mean_grid, axis=marginalize_axes)
                y_std_marginalized = np.sqrt(y_var_marginalized)

                ax.fill_between(x_grid_list[keep_axis], y1=y_marginalized-3*y_std_marginalized, y2=y_marginalized+3*y_std_marginalized, color=self.colors_std[j],alpha=0.2, label=f'{j} model mean $\pm 3\sigma$')
                ax.fill_between(x_grid_list[keep_axis], y1=y_marginalized-2*y_std_marginalized, y2=y_marginalized+2*y_std_marginalized, color=self.colors_std[j],alpha=0.2, label=f'{j} model mean $\pm 2\sigma$')
                ax.fill_between(x_grid_list[keep_axis], y1=y_marginalized-y_std_marginalized, y2=y_marginalized+y_std_marginalized, color=self.colors_std[j],alpha=0.2, label=f'{j} model mean $\pm 1\sigma$')
                ax.plot(x_grid_list[keep_axis],y_marginalized,color=self.colors_mean[j], label=f'{j} model mean')
                #ax.plot(x[:],y_data[:],'.',color="black", label="Data")
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin*0.95,ymax*1.05)
                #handles, labels = ax[j].get_legend_handles_labels()
                
                #order = [2,1,0]
                #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9, bbox_to_anchor=(0.665,1.),ncol=3)
                ax.set_ylabel('Data and Model Prediction')
                if j == (self.mf_model.nfidelities-1):
                    ax.set_xlabel(list(self.parameters.keys())[keep_axis])
            
            return fig
    
    def draw_yraw_contour(self, param_x, param_y, grid_steps=40, levels=20, cmap="viridis"):
            """Draw a filled contour plot of the high-fidelity (y_raw) prediction as function of two parameters.

            Parameters
            ----------
            param_x, param_y : str
                Names of the parameters to be shown on the horizontal and vertical axes, respectively.
            grid_steps : int, optional
                Number of grid points along each axis.
            levels : int, optional
                Number of contour levels.
            cmap : str, optional
                Colormap for the filled contours.
            """

            # Validate parameter names
            if param_x not in self.parameters or param_y not in self.parameters:
                raise ValueError(f"Parameters {param_x} and/or {param_y} not found in parameter list.")

            # Build grid for the two selected parameters
            x_vals = np.linspace(*self.parameters[param_x], grid_steps)
            y_vals = np.linspace(*self.parameters[param_y], grid_steps)
            Xg, Yg = np.meshgrid(x_vals, y_vals)

            # Prepare points to evaluate
            points = []
            for y in y_vals:
                for x in x_vals:
                    x_vec = self.x_fixed.copy()
                    x_vec[list(self.parameters.keys()).index(param_x)] = x
                    x_vec[list(self.parameters.keys()).index(param_y)] = y
                    points.append(x_vec)

            points = np.array(points)
            # Evaluate model at high fidelity (index 2 corresponds to y_raw)
            SPLIT = 1
            # Build list for convert_x_list_to_array
            x_list = [points, points, points]
            X_array = convert_x_list_to_array(x_list)
            mean_pred, _ = self.mf_model.model.predict(X_array[2*len(points):3*len(points)])
            Z = mean_pred.reshape(grid_steps, grid_steps)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            contour = ax.contourf(Xg, Yg, Z, levels=levels, cmap=cmap)
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label(r"Predicted $y_{raw}$")

            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_title(r"Predicted $y_{raw}$ contour")

            return fig
    
    

