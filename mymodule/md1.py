# %%
#--a good example of docstring can be found in--#
#--C:\Users\user\AppData\Local\Programs\Python\Python310\Lib\site-packages\tifffile\tifffile.py--#
r"""My 1st submodule 
    ---
    - class SpecTrum:
        * property:
            - flux
            - flux_masked
            - flux_nan
            - zscore_flux_masked
            - zscore_flux_nan
            - err
            - err_masked
            - err_nan
            - xaxis_obs
            - xaxis_rest
            - z
            - mask
            - mask_ind
            - dataset
        * setter:
            - update_mask(val)
            - reset_mask(val)
        * function:
            - copy(flux=None, mask=None, reset_mask=False)
            - 

    - class spectool: 
        * function:
            - xaxis_observed_frame(dataset) 
            - xaxis_rest_frame(xaxis,z)
            - wl_trans(name, ret_conven, data)
            - abs_fits(catalog)
            - find_abs_info(catalog,QSO_name,name_col)
            - zpick(dataset,included_line_list)
            - region_plot( 
                            arrangement=[],xlabel="",xlabel_fontsize=16, 
                            ylabel="",ylabel_fontsize=16,title="",title_fontsize=16, 
                            ylim=[-0.2,1.2],line=[], 
                            xdata=np.array([]),ydata=np.array([])
                        ) 
            - find_region(whole_xaxis,whole_yaxisdata,xmin,xmax) 
            - median_filter(data,median_interval) 
            - normalize(data,continuum)
            - get_velocity(z,line_wl,wl)
            - w2v(line_wl,wl)
            - zselect_vel_comp(data,range,increment,ec=None,rew=None,sf=None,rzsv=False)
            - rem_fal_sig(signal,width,rfs=False)
            - adaptive_median_filter(self, masked_flux, wlen)
            - mask_outlier(flux, threshold, mask=None)
            - mask_absorption(flux, threshold, mask=None)

    - class utils:
        * function:
            * orderpick(order_array, data) 
            * searchengine(Name,data, indices=True) 
            * lazy_regex_search(pattern, data, indices=True, ignorecase=True)
            * range_filter(data, range) 
            * find_global_max(xdata, ydata)
            * find_global_min(xdata, ydata)
            * find_duplicate(data, indices=False) 
            * compare_array(self, data1, data2) 
            * check_elements_val(array, val)
            * count_num_val(array, val)
            * prep_axvline(line_wl, line_list)
            * Nones(dim, dtype)
            * rem_inval_epd_bnd(smask, center, wlen)
            * rem_outliers(
                            threshold, data=None, zscore=None, 
                            wlen=5, ret_outlier=False, ret_zscore=False
                        )
            * get_middle_value(data)
            * test_ele_type(data, t)
            * true_mode(data, bins=1000, sigma=5)
            * save_obj(obj:object,path:str)

    - class multiaxes_graph:
        * function:
            * showplot() 
            * update() 

    - class twoaxes_graph:
        * function:
            * plot_graph(self,ylim=None,show=True,label=True,
            legend=True,labelstr=[],axvline=None) 
            * save_graph(path,file_name=None,dpi=100)

    - class myset:
        * function:
            * crossmatch(data1,data2) 
            * venn2_plot(data1, data2, name1='', name2='') 
    """
import functools
import copy
import pickle
import os
import re
import warnings
import statistics as st

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import astropy.io.ascii as ac
import scipy.ndimage as ni
import matplotlib_venn as venn

from datetime import datetime
from scipy import stats
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import r2_score

# %% 
class UnhandleError(Exception):
    pass
# %% 
class cry:
    def __init__(self, mask: np.ndarray, header : str) -> None:
        """pixel-wise criteria

        Parameters
        ----------
        mask : np.ndarray
            mask array of the `cry`
            `True` : wanted (show)
            `False` : unwanted

            In contrast, for `np.ma.masked_array`
            `True` : masked
            `False` : unmasked (show)

        header : str
            header of the `cry`
        """
        self._dc={}
        self._dc['mask'] = mask
        self._dc['header'] = header

    def __repr__(self) -> str:
        return f"header:{self._dc['header']} \n mask:{self._dc['mask']}"

    @property
    def mask(self) -> np.ndarray[np.bool_]:
        return self._dc['mask']

    @property
    def header(self) -> str:
        return self._dc['header']
    
    @property
    def masked_ind(self) -> np.ndarray:
        return np.where(self._dc['mask']==True)[0]
    
    # @property
    # def false_ind(self) ->np.ndarray:
    #     return np.where(self._dc['mask']==False)[0]

    @mask.setter
    def mask(self, val : np.ndarray[np.bool_]):
        self._dc['mask']=val

    @header.setter
    def header(self, val: str):
        self._dc['header']=val
# %% 
class SpecTrum:
    def __init__(
                self, 
                flux, 
                err, 
                z, 
                QSO_ID, 
                xaxis = None, 
                mask = False, 
                dataset = "k", 
                crop = False,
                line_wl = None,
                crys = None,
                peak = False,
            ):
        """the class of `SpecTrum` data retrieval

        Parameters
        ----------
        flux : 1D array, MaskedArray
            flux
        err : 1D array
            error
        z : float
            redshift of absorber
        QSO_ID : str
            QSO name
        xaxis : 1D array [default==None]
            observed wavelength,
            if the dataset is not provided (`dataset=None`)
        mask : bool, 1D array([bool...bool]), [default==False]
            additional mask,
            if the `flux` is `MaskedArray`,
            the final mask will be the union of `flux.mask` and `mask`
        dataset : str [default="k"]
            name of the dataset of the spectra, 
            KODIAQ, K, k: KODIAQ
        crop : bool [default=False]
            True: regional spectrum
            False : whole spectrum
        crys : numpy.ndarray([cry], dtype=list) [default=None]
            additional criteria,
            if `None` there will be no criteria
        peak : dict [default=False]
            peak parameter 

        Properties
        ----------
        flux
        flux_masked
        flux_nan
        zscore_flux_masked
        zscore_flux_nan
        err
        err_masked
        err_nan
        xaxis_obs
        xaxis_rest
        z
        mask
        mask_ind
        dataset

        Functions
        ---------

        Raises
        ------
        NameError
            if the name of the dataset cannot be found
        """
        # self._flux = flux
        self._flux = np.ma.array(flux)

        self._err=err
        self._z = z
        self._QSO_ID = QSO_ID
        # self._xaxis = xaxis

        # --old version--#
        # if isinstance(self._flux,np.ma.MaskedArray):
        #     self._mask=np.ma.mask_or(self._flux.mask,mask)
        # else:
        #     self._mask = mask
        # --old version--#

        self._mask=np.ma.mask_or(self._flux.mask,mask)

        self._crop = crop
        self._dataset = dataset
        
        if self._dataset in ('KODIAQ', 'K', 'k'):
            NAXIS1  =                33000 
            CRVAL1  =        3.59106460703
            CDELT1  =    1.44762400000E-05

            if not self._crop:
                self._xaxis_obs = 10**(CRVAL1+np.arange(NAXIS1)*CDELT1)
            elif xaxis is None:
                raise ValueError('This is a crop SpecTrum, please provide observed wavelength')
            else:
                self._xaxis_obs = xaxis
        elif self._dataset is None:
            self._xaxis_obs = xaxis
        else:
            raise ValueError(f"Dataset '{self._dataset}' is unavailable currently!")

        self._xaxis_rest = self._xaxis_obs/(1+self._z)

        self._line_wl=line_wl

        self._crys=np.array([],dtype=list)
        if crys is None:
            pass
        else:
            for i, ele in enumerate(crys):
                self._crys=np.append(self._crys, ele)

        self._peak=peak
        self._peak_dict={}
        if (self._peak != False): 
            if isinstance(self._peak, bool):
                if self._peak:
                    _flip=True
                    _prominence=0.0
                    _width=0.0
                    _rel_height=0.5
                    _height=None
                    _plateau_size=None
                    
            elif isinstance(self._peak, dict):
                try:
                    _flip=self._peak['flip']
                except KeyError:
                    _flip=True
            
                try:
                    _prominence=self._peak['prominence']
                except KeyError:
                    _prominence=0.0

                try:
                    _width=self._peak['width']
                except KeyError:
                    _width=0.0

                try:
                    _rel_height=self._peak['rel_height']
                except KeyError:
                    _rel_height=0.5

                try:
                    _height=self._peak['height']
                except KeyError:
                    _height=None

                try:
                    _plateau_size=self._peak['plateau_size']
                except KeyError:
                    _plateau_size=None

            if _flip:
                # detect valley
                f=-1
            else:
                # detect peak
                f=1

            peak_ind, peak_dict=signal.find_peaks(
                    f*self._flux.data, 
                    prominence=_prominence, 
                    width=_width, 
                    rel_height=_rel_height, 
                    height=_height, 
                    plateau_size=_plateau_size
                )
            
            # peak_dict['inds']=peak_ind
            self.append_ind2cry(peak_ind,'peak')
            # peak_dict['vels']=self.xaxis_velocity()[peak_ind]
            # peak_dict['skews']=np.diff(self._flux.data)[peak_ind-1]+np.diff(self._flux.data)[peak_ind]

            # self._peak_dict={}
            for ele in peak_dict.keys():
                l=np.ones_like(self._xaxis_obs,dtype=list)*np.nan
                l[peak_ind]=peak_dict[ele]
                self._peak_dict[ele]=l

    def __repr__(self)->str:
        return f'SpecTrum \n ID:{self._QSO_ID} \n crop:{self._crop} \n z={self._z} \n line wavelength={self._line_wl} \n x-axis (rest):{self._xaxis_rest[0]}~{self._xaxis_rest[-1]} \n x-axis (velocity):{self.xaxis_velocity()[0]}~{self.xaxis_velocity()[-1]} \n cry_headers:{self.cry_headers} \n peak_keys:{self.peak_keys}'

    # def update(self,mask):
    #     self._flux_masked = np.ma.array(self._flux, mask=self._mask, dtype=float)
    #     self._flux_nan = self._flux_masked.filled(float("nan"))
    #     self._zscore_flux_masked = np.ma.masked_invalid(sp.stats.zscore(self._flux_nan,nan_policy='omit'))

    # def check_dataset(func):
    #     @functools.wraps(func)
    #     def wrapper(self, *args, **kwargs):
    #         if self._dataset == "k":
    #             return func(self, *args, **kwargs)
    #         else:
    #             raise NameError(f"Dataset '{self._dataset}' is unavailable currently!")
    #     return wrapper

    @property
    def flux(self)->np.ndarray:
        """get the flux

        Returns
        -------
        np.ndarray
            flux array without mask
        """
        return self._flux.data
        
    # @flux.setter
    # def _flux(self,val):
    #     self._flux=val

    @property
    def flux_masked(self)->np.ma.MaskedArray:
        """the flux masked by `SpecTrum.mask`

        Returns
        -------
        np.ma.MaskedArray
            masked flux
        """
        return np.ma.array(self._flux, mask=self._mask, dtype=float)

    @property
    def flux_nan(self)->np.ndarray:
        """the flux with masked pixel replaced by `NaN`

        Returns
        -------
        np.ndarrray
            flux with `NaN`
        """
        self._flux_masked = np.ma.array(self._flux, mask=self._mask, dtype=float)
        return self._flux_masked.filled(float("nan"))

    @property
    def zscore_flux(self):
        self._flux_masked = np.ma.array(self._flux, mask=self._mask, dtype=float)
        self._flux_nan = self._flux_masked.filled(float("nan"))
        return np.ma.masked_invalid(stats.zscore(self._flux_nan,nan_policy='omit')).data

    @property
    def zscore_flux_masked(self):
        self._flux_masked = np.ma.array(self._flux, mask=self._mask, dtype=float)
        self._flux_nan = self._flux_masked.filled(float("nan"))
        return np.ma.masked_invalid(stats.zscore(self._flux_nan,nan_policy='omit'))

    @property
    def zscore_flux_nan(self):
        self._flux_masked = np.ma.array(self._flux, mask=self._mask, dtype=float)
        self._flux_nan = self._flux_masked.filled(float("nan"))
        self._zscore_flux_masked = np.ma.masked_invalid(stats.zscore(self._flux_nan,nan_policy='omit'))
        return self._zscore_flux_masked.filled(float("nan"))

    @property
    def err(self):
        return self._err
    
    @property
    def err_masked(self):
        return np.ma.array(self._err, mask=self._mask, dtype=float)
    
    @property
    def err_nan(self):
        self._err_masked=np.ma.array(self._err, mask=self._mask, dtype=float)
        return self._err_masked.filled(float('nan'))

    @property
    def xaxis_obs_masked(self)->np.ma.MaskedArray:
        """get the masked xaxis in observed frame, the mask is `SpecTrum.mask`

        Returns
        -------
        np.ma.MaskedArray
            masked xaxis in observed frame
        """
        return np.ma.array(self._xaxis_obs,mask=self._mask)

    @property
    def xaxis_obs(self)->np.ndarray:
        """get the xaxis in observed frame

        Returns
        -------
        np.ndarray
            xaxis in observed frame
        """
        return self._xaxis_obs

    @property
    def xaxis_rest_masked(self)->np.ma.MaskedArray:
        """get the masked xaxis in rest frame

        Returns
        -------
        np.ma.MaskedArray
            masked xaxis in rest frame
        """
        return np.ma.array(self._xaxis_rest,mask=self._mask)

    @property
    def xaxis_rest(self)->np.ndarray:
        """get the xaxis in rest frame

        Returns
        -------
        np.ndarray
            xaxis in rest frame
        """
        return self._xaxis_rest

    @property
    def z(self):
        return self._z

    @property
    def QSO_ID(self)->str:
        return self._QSO_ID

    @property
    def mask(self):
        return self._mask

    @property
    def masked_ind(self):
        return np.where(self._mask == True)[0]

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def crop(self):
        return self._crop
    
    @property
    def line_wl(self):
        return self._line_wl
    
    @property
    def crys(self):
        return self._crys
    
    @property
    def cry_headers(self):
        """the list of headers of `cry` of the `SpecTrum`

        Returns
        -------
        list
            list of headers
        """
        return [i.header for i in self._crys]
    
    @property
    def peak(self):
        return self._peak
    
    @property
    def peak_dict(self):
        try:
            return self._peak_dict
        except AttributeError:
            raise ValueError(f'`peak` is `{self._peak}`, `peak_dict` doesn\'t exist.')
        
    @property
    def peak_keys(self):
        return [key for key in self._peak_dict.keys()]

    def _makesure_size(func):
        """implicit function to makesure the size of `mask`
        """
        @functools.wraps(func)
        def wrapper(self,*args,**kwargs):
            val=args[0] # The first argument corresponds to 'val'
            if isinstance(val, (list , np.ndarray)):
                if isinstance(val[0],(bool,np.bool_)): # use the first element to distinguish
                    if len(val) == 1:
                        # if use a single bool in the list, the bool will be broadcasted to all data
                        pass
                    elif len(val) == len(self._flux):
                        pass
                    else:
                        raise np.ma.MaskError(
                            f"Mask and flux not compatible: flux size is {len(self._flux)}, mask size is {len(val)}."
                        )
                elif isinstance(val[0], cry): # use the first element to distinguish
                    for ele in val:
                        if len(ele.mask)!=len(self._flux):
                            raise np.ma.MaskError(
                                f"Mask and flux not compatible: flux size is {len(self._flux)}, mask size of header \'{ele.header}\' is {len(ele.mask)}."
                            )

            elif isinstance(val, (bool, np.bool_)):
                pass

            elif isinstance(val, cry):
                if isinstance(val.mask, np.ndarray):
                    if len(val.mask) == len(self._flux):
                        pass
                    else:
                        raise np.ma.MaskError(
                            f"Mask and flux not compatible: flux size is {len(self._flux)}, mask size is {len(val.mask)}."
                        )
                    
                elif isinstance(val.mask, np.bool_):
                    pass
                
            else:
                raise np.ma.MaskError(
                    f"Use bool or [bool] or [bool,...,bool] (the same size as flux), not {val}"
                )
            return func(self,*args,**kwargs)
        return wrapper

    @mask.setter
    @_makesure_size
    def update_mask(self, val:np.ndarray):
        """update the mask,
        the final mask will be the union of original mask and this mask

        Parameters
        ----------
        val : 1D array ([bool...bool])
            new mask
        """
        self._mask = np.ma.mask_or(self._mask,val)

    @mask.setter
    @_makesure_size
    def reset_mask(self, val):
        """reset the mask,
        the orignal mask will be replaced with this mask

        Parameters
        ----------
        val : 1D array ([bool...bool])
            new mask
        """
        self._mask=val

    # @crys.setter
    @_makesure_size
    def append_cry(self, val : cry | list[cry]):
        """append a new `cry` or a list of new `cry` 

        Parameters
        ----------
        val : cry | list[cry]
            new `cry` to be appended

        Raises
        ------
        ValueError
            Header `val.header` is repeating, please rename the header.
        """
        # maskesure no repeating header
        # 1 `cry`
        if isinstance(val,cry):
            for i in self._crys:
                if i.header==val.header:
                    raise ValueError(
                        f'Header \'{val.header}\' is repeating, please rename the header.'
                        )
        # a list of `cry`
        else:
            for ele in val:
                for i in self._crys:
                    if i.header==ele.header:
                        raise ValueError(
                            f'Header \'{ele.header}\' is repeating, please rename the header.'
                        )
            
        self._crys = np.append(self._crys, val)

    def append_ind2cry(self, ind: np.ndarray, header:str):
        # self._crys=np.append(self._crys, spectool().ind2cry(self,ind,header))
        """create a `cry` from an array of indices

        Parameters
        ----------
        ind : np.ndarray
            indices to create cry
        header : str
            the header of the cry
        """
        self.append_cry(spectool().ind2cry(self,ind,header))

    # def append_ind2cry_new(self, ind: np.ndarray, header):
    #     self.append_cry(spectool().ind2cry_new(self,ind,header))

    def append_mask2cry(self, mask:np.ndarray[np.bool_], old_header:str, new_header:str):
        # mask: True: unwanted, False: wanted
        """mask an old `cry` to create new `cry`

        Parameters
        ----------
        mask : np.ndarray[np.bool_]
            mask
        old_header : str
            the header of old cry
        new_header : str
            the header of new cry
        """
        self.append_ind2cry(
            np.ma.array(
                self.cry(old_header).masked_ind, mask=mask
            ).compressed(),
            new_header,
        )

    def intersect_cry(self, cry_header:str, cry_header1:str, new_header:str):
        """get the intersect of `cry`

        Parameters
        ----------
        cry_header : str
            the first header of `cry`
        cry_header1 : str
            the second header of `cry`
        new_header : str
            the header of the new `cry`
        """
        self.append_cry(
            cry(
                spectool().cry_and(
                    self.cry(cry_header).mask, 
                    self.cry(cry_header1).mask,
                ),
                new_header,
            )
        )

    def union_cry(self, cry_header:str, cry_header1:str, new_header:str):
        """get the union of `cry`

        Parameters
        ----------
        cry_header : str
            the first header of `cry`
        cry_header1 : str
            the second header of `cry`
        new_header : str
            the header of the new `cry`
        """
        self.append_cry(
            cry(
                spectool().cry_or(
                    self.cry(cry_header).mask, 
                    self.cry(cry_header1).mask,
                ),
                new_header,
            )
        )

    # @crys.deleter
    # def delete_cry_old(self, val : str | list[str]):
    #     if isinstance(val,list):
    #         for ele in val:
    #             l=[i for i in self._crys if i.header==ele]
    #             if len(l)==0:    # if the header is not found.
    #                 raise ValueError(f'The cry with header \'{ele}\' is not found.')
    #             else:
    #                 ind=np.where(self._crys==l)[0][0]
    #             self._crys=np.delete(self._crys, ind)
    #     else:
    #         l=[i for i in self._crys if i.header==val]
    #         if len(l)==0:    # if the header is not found.
    #             raise ValueError(f'The cry with header \'{val}\' is not found.')
    #         else:
    #             ind=np.where(self._crys==l)[0][0]
    #         self._crys=np.delete(self._crys, ind)

    def delete_cry(self, ind_or_hd: int| str| list[str]):
        """delete the `cry`

        Parameters
        ----------
        ind_or_hd : int | str | list[str]
            a or list of index or header of the cry
        """
        if isinstance(ind_or_hd,int):
            self._crys=np.delete(self._crys,ind_or_hd)
        elif isinstance(ind_or_hd,str):
            self._crys=np.delete(self._crys,self.cry(ind_or_hd,True)[1])
        elif isinstance(ind_or_hd, list):
            for ele in ind_or_hd:
                self._crys=np.delete(self._crys,self.cry(ele,True)[1])


    def copy(self, flux=None, err=None, mask=None, reset_mask=False):
        """deepcopy of ancestor, inherit its `err`, `z`, `QSO_ID`, and `xaxis`

        Parameters
        ----------
        flux : 1D array, [default=None]
            new flux,
            if `None` inherit from ancestor
        mask : 1D array, [default=None]
            additional mask,
            the final mask will be the union of `flux.mask` and `mask`
        reset_mask : bool, [default=None]
            whether to reset mask

        Returns
        -------
        SpecTrum : class
            deepcopy of `SpecTrum`
        """
        _self=copy.deepcopy(self)

        if flux is not None:
            _self._flux=flux

        if (mask is not None) & (reset_mask==False):
            _self.update_mask=mask
        elif (mask is not None) & (reset_mask==True):
            _self.reset_mask=mask
        
        if err is not None:
            _self._err=err

        return _self
    
    def xaxis_velocity_masked(self, line_wl = None) -> np.ma.MaskedArray:
        """wavelength to velocity centering "a specific line"

        Parameters
        ----------
        line_wl : float | int [default=None]
            rest wavelength of "the specific line" in Å,
            if `None` use the line_wl provided of `SpecTrum`

        Returns
        -------
        vel : MaskedArray
            xaxis in velocity space centering `line_wl`
        """
        if line_wl is not None:
            _line_wl=line_wl
        else:
            _line_wl=self._line_wl
        # the formulae used here are base on `VoigtFit.container.regions.get_velocity()`
        lcen=_line_wl*(self._z+1) # lcen : line center
        vel=(self._xaxis_obs-lcen)/lcen*299792.458 # km/s
        return np.ma.array(vel,mask=self._mask)

    def xaxis_velocity(self, line_wl = None ) -> np.ndarray:
        """wavelength to velocity centering "a specific line"

        Parameters
        ----------
        line_wl : float | int [default=None]
            rest wavelength of "the specific line" in Å,
            if `None` use the line_wl provided of `SpecTrum`

        Returns
        -------
        vel : 1D array
            xaxis in velocity space centering `line_wl`
        """
        # the formulae used here are base on `VoigtFit.container.regions.get_velocity()`
        if line_wl is not None:
            _line_wl=line_wl
        else:
            _line_wl=self._line_wl

        lcen=_line_wl*(self._z+1) # lcen : line center
        vel=(self._xaxis_obs-lcen)/lcen*299792.458 # km/s

        return vel
    
    def cry(self, ind_or_hd: int | str, ret_ind=False)->cry:
        """retrieve the `cry` with its index or header

        Parameters
        ----------
        ind_or_hd : int | str
            index or header
        ret_ind : bool, [default=False]
            whether to return the index

        Returns
        -------
        cry
            wanted cry

        Raises
        ------
        ValueError
            The cry with header `ind_or_hd` is not found.
        ValueError
            `ind_or_hd` should be either an `int` or `str`
        """
        if isinstance(ind_or_hd,int):
            _ind_or_hd=ind_or_hd
        elif isinstance(ind_or_hd, str):
            l=[i for i in self._crys if i.header==ind_or_hd]
            if len(l)==0:    # if the header is not found.
                raise ValueError(f'The cry with header \'{ind_or_hd}\' is not found.')
            _ind_or_hd=np.where(self._crys==l)[0][0]
        else:
            raise ValueError(f'`ind_or_hd` should be either an `int` or `str`!')

        if ret_ind:
            return self._crys[_ind_or_hd], _ind_or_hd
        else:
            return self._crys[_ind_or_hd]   

    def xaxis_cry(self, ind_or_hd: int | str , frame='vel',line_wl=None)->np.ma.MaskedArray: 
        """ display the xaxis based on given `cry`

        Parameters
        ----------
        ind_or_hd : int | str
            index of cry or header of cry
        frame : str [default='vel']
            frame
            'velocity' or 'vel' : velocity frame
            'rest' : rest frame
            'obs' : observed frame
        line_wl : float | int [default=None]
            wavelength of line that want to be the centroid
            if `None`, use the original `line_wl` of `SpecTrum`

        Returns
        -------
        np.ma.MaskedArray
            True : show
            False : not show

        Raises
        ------
        ValueError
            `frame` is other than "velovity", "vel", "rest" or "obs"
        """
        if frame in ('velocity', 'vel'):
            if line_wl is not None:
                return np.ma.array(self.xaxis_velocity(line_wl), mask=~self.cry(ind_or_hd).mask)
            else:
                # raise ValueError('Please provide `line_wl`.')
                return np.ma.array(self.xaxis_velocity(), mask=~self.cry(ind_or_hd).mask)
        elif frame=='rest':
            return np.ma.array(self._xaxis_rest, mask=~self.cry(ind_or_hd).mask)
        elif frame=='obs':
            return np.ma.array(self._xaxis_obs, mask=~self.cry(ind_or_hd).mask)

        else:
            raise ValueError(f'`frame` accept "velovity", "vel", "rest" and "obs", not `{frame}`.')
        
    def flux_cry(self, cry_header : str)->np.ma.MaskedArray:
        # return self._flux.data[self.cry(cry_header).masked_ind]
        return np.ma.array(self._flux.data, mask=~self.cry(cry_header).mask)

    def peak_cry(self, peak_key:str, cry_header:str)->np.ma.MaskedArray:
        # return self._peak_dict[peak_key][self.cry(cry_header).mask]
        return np.ma.array(self._peak_dict[peak_key], mask=~self.cry(cry_header).mask)
    
    def append_peak_dict(self, new_key: str, data: np.ndarray , cry_header=None)->None:
        _data=np.ones_like(self._xaxis_obs,dtype=list)*np.nan
        if cry_header is None:
            _data=data
        else:
            _data[self.cry(cry_header).masked_ind]=data

        self._peak_dict[new_key]=_data

# %% create VelCom
class VelCom:
    def __init__(self, key: np.ndarray, val: np.ndarray, vctype: str, id: int) -> None:
        self._key = key
        self._val = val
        self._vctype = vctype # type of velocity component {'peak','knee','pwf'}
        self._dt = {key: value for key, value in zip(self._key, self._val)}
        self._id = id
        # self._tie = set()
        self._tie = np.array([], dtype=int)

    @property
    def keys(self):
        # return np.array(list(self._dt.keys()))
        return self._key

    @property
    def values(self):
        return self._val

    @property
    def vc_dict(self):
        self._dt = {key: value for key, value in zip(self._key, self._val)}
        return self._dt

    @property
    def vctype(self):
        return self._vctype

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val

    def __getattr__(self, key):
        if key in self._dt:
            return self._dt[key]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def append(self, key, val):
        self._key = np.append(self._key, key)
        self._val = np.append(self._val, val)
        self._dt[key] = val

    @property
    def tie(self):
        return np.asarray(list(set(self._tie)), dtype=int)

    def add_tie(self, val):
        _val = np.asarray(val, dtype=int)
        self._tie = np.append(self._tie, _val)

# %%
class spectool:
    r"""class spectool: 
        * function:
            - load_spectrum()
            - xaxis_observed_frame(dataset) 
            - xaxis_rest_frame(xaxis,z)
            - wl_trans(name, ret_conven, data)
            - abs_fits(catalog)
            - find_abs_info(catalog,QSO_name,name_col)
            - zpick(dataset,included_line_list)
            - region_plot( 
                            arrangement=[],xlabel="",xlabel_fontsize=16, 
                            ylabel="",ylabel_fontsize=16,title="",title_fontsize=16, 
                            ylim=[-0.2,1.2],line=[], 
                            xdata=np.array([]),ydata=np.array([])
                        ) 
            - find_region(whole_xaxis,whole_yaxisdata,xmin,xmax) 
            - median_filter(data,median_interval) 
            - normalize(data,continuum)
            - get_velocity(z,line_wl,wl)
            - w2v(line_wl,wl)
            - zselect_vel_comp(data,range,increment,ec=None,rew=None,sf=None,rzsv=False)
            - rem_fal_sig(signal,width,rfs=False)
            - adaptive_median_filter(self, masked_flux, wlen)
            - mask_outlier(flux, threshold, mask=None)
            - mask_absorption(flux, threshold, mask=None)
    """
    def __init__(self):
        pass

    def load_spectrum(self, QSO_ID, catalog: str, dataset='k')->tuple[np.ndarray,np.ndarray,float,np.ndarray]:
        """load the `.spec` file

        Parameters
        ----------
        QSO_ID : str
            ID of QSO
        catalog : str
            catalog, "Mg" or "C"
        dataset : str, optional
            _description_, by default 'k'

        Returns
        -------
        `tuple(np.ndarray,np.ndarray,float,np.ndarray)`
            

        Raises
        ------
        ValueError
            _description_
        """
        if dataset in ('KODIAQ', 'K', 'k'):
            wl, flux, err = np.loadtxt(
                r"D:\University_Course\Project\summer_program_2022\KODIAQ_specfile_neu\%s.spec"
                % QSO_ID,
                unpack=True,
            )
            # z = zabs_array[ind]
            z = spectool().find_z(QSO_ID,catalog)
        else:
            raise ValueError(f'`{dataset}` is not avaliable currently.')
        return flux, err, z, wl
    
    def load_QSO_ID(self,catalog: str)->np.ndarray:
        """load the QSO_ID array

        Parameters
        ----------
        catalog : str
            'Mg' : Mg catalog
            'C' : C catalog
            'union' : union of Mg and C catalog 

        Returns
        -------
        QSO_ID : np.ndarray
            QSO_ID array

        Raises
        ------
        ValueError
            the catalog is unavailable
        """
        if catalog=='C':
            return np.load(
                # ---- old catalog (contain out_of_z) ----#
                # r"D:\University_Course\Project\summer_program_2022\KODIAQ_CIV_QSO_Name_CIV1548.npy",
                # ---- old catalog (contain out_of_z) ----#
                r"D:\University_Course\Project\summer_program_2022\flow\catalog\C_catalog.npy",
                allow_pickle=True,
            )
        elif catalog=='Mg':
            return np.load(
                # ---- old catalog (contain out_of_z) ----#
                # r"D:\University_Course\Project\summer_program_2022\S12_S7_QSO_Name_MgII2796_ZABS135_Unique.npy",
                # ---- old catalog (contain out_of_z) ----#
                r"D:\University_Course\Project\summer_program_2022\flow\catalog\Mg_catalog.npy",
                allow_pickle=True,
            )
        elif catalog=='union':
            return np.load(
                r"D:\University_Course\Project\summer_program_2022\flow\catalog\union_catalog.npy",
                allow_pickle=True,
            )
        else:
            raise ValueError(f"Catalog '{catalog}' is unavailable currently!")
        
    #---has been integrate into `SpecTrum`, still perserved for now---#
    def xaxis_observed_frame(self,dataset='k'):
        """wavelength in observed frame of KODIAQ

        Parameters
        ----------
        dataset : str, optional [default='k']
            dataset for this function, only KODIAQ for now
            type:'KODIAQ','K','k' for KODIAQ

        Returns
        -------
        xaxis : 1D array
            data of xaxis
        """
        if dataset in ('KODIAQ', 'K', 'k'):
            NAXIS1  =                33000 
            CRVAL1  =        3.59106460703
            CDELT1  =    1.44762400000E-05
        else:
            raise ValueError(f'`{dataset}` is not avaliable currently.')

        # --old version--# 
        # xaxis=np.zeros(NAXIS1)
        # for t in range(NAXIS1) :
        #     xaxis[t]=10**(CRVAL1+t*CDELT1)
        # --old version--# 

        xaxis = np.arange(NAXIS1)
        _xaxis = 10**(CRVAL1+xaxis*CDELT1)
        return _xaxis
    #---has been integrate into `SpecTrum`, still perserved for now---#
    
    #---has been integrated into `SpecTrum`, still perserved for now---#
    def xaxis_rest_frame(self,z,xaxis=None,dataset='k'):
        """wavelength in rest frame of KODIAQ

        Parameters
        ----------
        z : float
            redshift of the absorber
        xaxis : 1D array, optional [default=None]
            xaxis in rest frame
        dataset : str, optional [default='k']
            dataset for this function, only KODIAQ for now
            type:'KODIAQ','K','k' for KODIAQ

        Returns
        -------
        rest_xaxis : 1D array
            xaxis in rest frame
        """
        if (xaxis is None) & (dataset=='k'):
            rest_xaxis=spectool().xaxis_observed_frame()/(1+z)
        else:
            rest_xaxis=xaxis/(1+z)
            
        return rest_xaxis
    #---has been integrate into `SpecTrum`, still perserved for now---#

    def wl_trans(self, name: list | str, ret_conven=False, data='VoigtFit'):
        """find the wavelength of transition with `VoigtFit.static.linelist`

        Parameters
        ----------
        name : list[str] | str
            name of the transition,
            such as, "MgII_2796".

            If you know the atom, but don't know the transition,
            or you just want to search how many transitions does the atom correspond to,
            you can type the name of the atom,
            such as "MgII".

            If you only know the wavelength but don't know the atom,
            you can type the wavelength,
            such as "2796". 
            
            For the later 2 cases, you can set `ret_conven=True`,
            in order to find the complete name of the transition.

        ret_conven : bool, optional [default=False]
            whether to return a list of conventional notation of transition,
            such as "MgII_2796".

        data : str
            data of transition to be searched,
            currently, only `VoigtFit.static.linelist` is available

        Returns
        -------
        lw : 1D array | float
            wavelength of the transition to 4 decimal places,   \n
            such as `2796.3522`,    \n
            return `float` if `name` is `str`

        convention : 1D array
            astronomy convention of transition, \n
            returned when `ret_conven=True`,    \n
            return `str` if `name` is `str`

        Raises
        ------
        NameError
            if the line isn't found in the dataset
        """
        #--to be refined with `Regex`--#
        # pattern= re.compile("VoigtFit",re.IGNORECASE)
        # match=pattern.search(data)
        # match=re.findall("voigtfit", data, re.IGNORECASE)
        #--to be refined with `Regex`--#

        if data == "VoigtFit" :
            path=r"C:\Users\user\AppData\Local\Programs\Python\Python310\Lib\site-packages\VoigtFit\static\linelist.dat"
        else:
            raise NameError("No database available")
        
        if isinstance(name,list):
            _name=name
        else:
            _name=[name]

        with open(path,"r") as file : # open file and close it after using
            lines=file.readlines() # read all lines
            line_indices=[] # create a list to store the index of the line
            for i, l in enumerate(lines):
                for n in _name:
                    if n in l:
                        line_indices.append(i)

            if len(line_indices) >0:
                line_splited=np.zeros(len(line_indices),dtype=object)
                lw=np.zeros_like(line_splited)
                con=np.zeros_like(line_splited)
                for i, line_index in enumerate(line_indices):
                    line_splited[i]=lines[line_index].split()

                    lw[i]=float(line_splited[i][2])
                    con[i]=line_splited[i][0]
            else:
                raise NameError("No line has been found")

        # test={}
        if isinstance(name,list):
            # test['a']=0
            _lw=lw
            _con=con
        else:
            # test['a']=1
            _lw=lw[0]
            _con=con[0]

        if ret_conven is True: 
            # test['b']=0   
            # return _lw, _con, test
            return _lw, _con
        else:
            # test['b']=1
            # return _lw, test
            return _lw

    def abs_fits(self, catalog):
        raise AttributeError('`abs_fits()` has change to `abs_catalog()`')

    # --new version of abs_fits()--#       
    def abs_catalog(self,catalog):
        """return the FITS file of the absorber catalog

        Parameters
        ----------
        catalog : str
            'Mg': 'Union_K_S12_S7_RA&Dec_crossmatch_MgII2796_ZABS135.fits'
            'C': 'CIV_cooksey_with_magnitude.fits' 

        Examples
        ----------
        >>> abs_fits('Mg').columns.names
        >>> ['Name_1',
            'RAJ2000_1',
            ...
            'Separation_1']

        Returns
        -------
        HDUList: HDUList
            HDUList of catalog
        """
        if catalog=='Mg':
            HDUList=pf.open(r"D:\University_Course\Project\summer_program_2022\KODIAQ_SDSS_crossmatch\Union_K_S12_S7_RA&Dec_crossmatch_MgII2796_ZABS135.fits")
        elif catalog=='C':
            HDUList=pf.open(r"D:\University_Course\Project\summer_program_2022\KODIAQ_CIV_crossmatch_CIV1548")
        else:
            raise KeyError(f'There is no {catalog} catalog! Try \'Mg\' or \'C\'. ')
        return HDUList
    # --new version of abs_fits()--#

    def find_abs_info(self,catalog: str, QSO_name: str, name_col: str)->list:
        r"""find the information in the absorber catalog

        Parameters
        ----------
        catalog : str
            'Mg' or 'C'
        QSO_name : str
            name of the QSO
        name_col : str 
            name of the column, in regex pattern

        Returns
        -------
        info : list
            a list of the results of query

        Raises
        ------
        KeyError
            if there is no result
        KeyError
            if there are more than 1 results

        Notes
        ------
        use raw string if there are '\\' contained 

        Examples
        ------
        use raw string & word boundary "\\b" 
        to distinguish 'REW_MGII_2796_1' from 'ERR_REW_MGII_2796_1'
        >>> spectool().find_abs_info("Mg","J001602-001225", r"\brew_mgii_2796_1\b")
        """
        fits_file=spectool().abs_catalog(catalog)
        col_name_QSO_name=utils().lazy_regex_search("name", fits_file[1].data.columns.names, 0, 1)[0] # column name of QSO_name
        QSO_ind=utils().searchengine(QSO_name,fits_file[1].data[col_name_QSO_name],1)[1][0]

        if isinstance(name_col,list):
            pass
        else:
            name_col=[name_col]

        result=[]

        for i in name_col:

            _name_col=utils().lazy_regex_search(i, fits_file[1].data.columns.names, 0, 1)

            if len(_name_col)==0:
                raise KeyError(fr'There is {len(_name_col)} result matching {i}. \n Try another.')            
            elif len(_name_col)!=1:
                raise KeyError(fr'There are {len(_name_col)} results matching {i}. \n Please be specific. \n {_name_col}')
            else:
                result.append(fits_file[1].data[_name_col[0]][QSO_ind])
                
        return result

    def find_z(self, QSO_ID: str, catalog: str)->float:
        """find the redshift of the absorber in the catalog
            a quick version of 
            >>> md.spectool().find_abs_info("Mg",QSO_ID,'ZABS_1')
        
        Parameters
        ----------
        QSO_ID : str
            ID of QSO
        catalog : str
            'Mg': Mg catalog
            'C': C catalog

        Returns
        -------
        redshift : float
            redshift of absorber
        """

        fits=spectool().abs_catalog(catalog)[1]

        ID_col_name=utils().lazy_regex_search('name',fits.columns.names)[0][0]
        z_col_name=utils().lazy_regex_search('zabs',fits.columns.names)[0][0]

        ID_ind=utils().searchengine(QSO_ID,fits.data[ID_col_name])[1][0]

        return fits.data[z_col_name][ID_ind]

    #---to be merged into `utils`---#
    def zpick(self,included_line_list,dataset='k'):
        r"""choose the spectra with Z_absorber (only KODIAQ for now) within a given range of redshift, 
        currently only served for lines which should be included.
        (the version for the lines which should be excluded is upcoming)

        Parameters
        ----------
        dataset: string
            dataset for this function, only KODIAQ for now.
            type:'KODIAQ','K','k' for KODIAQ

        included_line_list : list or 1D array
            a list of line of interest

        Returns
        -------
        included_zrange: ndarray
            np.array([z_min,z_max])
        """
        if dataset in ('KODIAQ', 'K', 'k'):
            NAXIS1=                33000
            CRVAL1=        3.59106460703                                              
            CDELT1=    1.44762400000E-05

        wl_min=10**(CRVAL1)
        wl_max=10**(CRVAL1+NAXIS1*CDELT1)
        def _zpick(line_list):
            if isinstance(line_list,np.ndarray):
                pass
            else:
                line_list=np.asarray(line_list)

            z_lowerbound=wl_min*1/line_list-1
            z_upperbound=wl_max*1/line_list-1
            z_min=np.amax(z_lowerbound)
            z_max=np.amin(z_upperbound)
            return np.array([z_min, z_max])
        included_zrange=_zpick(included_line_list)
        # excluded_zrange=zpick()
        return included_zrange
    #---to be merged into `utils`---#

    #---to be decomission---#
    def region_plot(
        self,arrangement=[],xlabel="",xlabel_fontsize=16,
        ylabel="",ylabel_fontsize=16,title="",title_fontsize=16,
        ylim=[-0.2,1.2],line=[],
        xdata=np.array([]),ydata=np.array([])):
        r""" plot absorption region with line

        Parameters
        ----------
        arrangement: list
            [row_num,column_num,plot_order]

        xlabel: str

        xlabel_fontsize: int

        ylabel: str

        ylabel_fontsize: int

        title: str

        title_fontsize: int

        ylim: list
            [lower_limit,upper_limit]

        line: list [default=[]]
            list of lines need to plot on the region
                
        xdata: 1D ndarray

        ydata: 1D ndarray
        """
        self=plt.subplot(arrangement[0],arrangement[1],arrangement[2])
        self.set_xlabel(xlabel=xlabel,fontsize=xlabel_fontsize)
        self.set_ylabel(ylabel=ylabel,fontsize=ylabel_fontsize)
        self.set_title(title,fontsize=title_fontsize)
        self.set_ylim(ylim[0],ylim[1])
    
        for i in line:
            self.axvline(x=i,linestyle='--',color=((i-int(i))/13*9,(i-int(i))/9,(i-int(i))/11))

        self.plot(xdata,ydata)        
    #---to be decommissioned---#

    #---to be merged into `SpecTrum`---#
    def find_region(self,whole_xaxis,whole_yaxisdata,xmin,xmax):
        """ find the given region of the spectra
            ------
            Parameters
            whole_xaxis: ndarray
                whole data of xaxis (redshifted or not)
            
            whole_yaxis: ndarray
                whole data of yaxis (normalized or not)
    
            xmin: float
                left side of wavelength span
                
            xmax: float
                right side of wavelength span

            ------
            Returns
            [region_xaxis,region_yaxisdata]    
            """
        #--after i have a good continuum pipeline, update with ylim, xlim when rendering the graph--#
        region_xaxis=np.zeros(0)
        region_xaxis_order=np.zeros(0,dtype=int)
        region_yaxisdata=np.zeros(0)

        for i in whole_xaxis:
            if xmin<=i<=xmax:
                region_xaxis=np.append(region_xaxis,i)
                region_xaxis_order=np.append(region_xaxis_order,np.where(whole_xaxis==i))
            else:
                pass

        for ii in region_xaxis_order:
            region_yaxisdata=np.append(region_yaxisdata,whole_yaxisdata[ii])
        #--after i have a good continuum pipeline, update with ylim, xlim when rendering the graph--#

        return [region_xaxis,region_yaxisdata]

    def median_filter(self,data,median_interval):
        """ simplified version of `scipy.ndimage.median_filter()` 
        
            data: ndarray
                data to be medianfiltered
                
            median_interval: int
                number of data points in the interval
            --------
            # Return
            medianfiltered data
            """
        data_medianfiltered=ni.median_filter(data,median_interval)
        return data_medianfiltered
    
    def normalize(self,data,continuum):
        r""" normalization

            Parameters
            ----------
            data: 1D ndarray, 1D MaskedArray
                data to be normalized
                
            continuum:1D ndarray, 1D MaskedArray
                continuum data (same size as data) 

            Returns
            -------
            data_normalized: 1D array
                normalized data
            """
        #--decorator practice, correct by ChatGPT--#
        # def makesure_MaskedArray(func):
        #     @functools.wraps(func)
        #     def wrapper(*args, **kwargs):
        #         result = func(*args, **kwargs)
        #         return np.ma.array(result)
        #     return wrapper

        # @makesure_MaskedArray
        # def get_data(data):
        #     return data
        
        # return get_data(data)/get_data(continuum)
        #--decorator practice, correct by ChatGPT--# 

        #--the most concise version--#
        return np.ma.array(data)/np.ma.array(continuum)
        #--the most concise version--#

    #---to be merged into `SpecTrum`---#
    def get_velocity(self,z,line_wl,wl):
        """get the xaxis in velocity space centering "a specific line"

        Parameters
        ----------
        z : float, list, 1D array 
            systemtic redshift
        line_wl : float
            rest wavelength of "the specific line" in Å
        wl : 1D array
            wavelength of the spectrum

        Returns
        -------
        vel : 1D array
            velocity that is transformed from wavelength in km/s
        """
        # z: list or 1D array
        # lcen : line center
        # wl : wavelength

        if isinstance(z,float):
            z=np.array([z])
        elif isinstance(z,list):
            z=np.asarray(z)
        else:
            pass

        # the formulae used here are base on `VoigtFit.container.regions.get_velocity()`
        lcen=line_wl*(z+1)
        vel=(wl-lcen)/lcen*299792.458 # km/s

        # --my version (incorrect)--#
        # vel=(wl-lcen)/line_wl*299792.458 # km/s
        # --my version (incorrect)--#
        return vel
    #---to be merged into `SpecTrum`---#    

    def get_velocity_mono(self,line_wl,wl):
        raise AttributeError('`get_velocity_mono()` has changed to `w2v()`')

    # --new version of `get_velocity_mono()`--#
    def w2v(self,line_wl,wl):
        """transfer a number of line 
        from wavelength space to velocity space

        Parameters
        ----------
        line_wl : 
            wavelength of transition
        wl : list, 1D array
            list of rest wavelemgth to transfer into velocity

        Returns
        -------
        1D array
            _description_
        """
        _wl=np.asarray(wl)
        vel=(_wl-line_wl)/line_wl*299792.458
        return vel
    # --new version of `get_velocity_mono()`--#
    
    def v2w(self, line_wl, v):
        _v=np.asarray(v)
        wl=line_wl+line_wl*_v/299792.458
        return wl

    def zselect_vel_comp(self,data,range,increment,ec=None,rew=None,sf=None,rzsv=False): 
        r"""select a given number of velocity components that in a specific z scores range

        Parameters
        ----------
        data : 1D array
            array to select from
        range : float, list
            the range of zscore that the absorption lines should satisfy
            float: the upper bound, will be transform to `[float(-'inf'),range]`
            list: [lower bound, upper bound]
        increment : float
            How many should the upper bound of range increase.
            determine the sensitivity of algorithm.
            recommand `0.2`
        ec : int, float, optional [default==None]
            expected number of components
            if not given, then `ec=rew*sf`
        rew : float, optional [default==None]
            rest equival width of absorption line,
            if `ec` isn't given, must mention
        sf : int, float, optional [default==None]
            scaling factor of `rew`
            if `ec` isn't given, must mention
        rzsv: bool, optional [default==False]
            whether to return `zselected_val`
            True: return
            False: not return
        
        Returns
        -------
        zselected_ind: 1D array
            the indices of selected value in `data`
        zselected_val: 1D array
            the selected value in `data`
        """
        zscore=stats.zscore(data,nan_policy="omit") # input an array of y-axis data

        if isinstance(range,list):
            pass
        else:   # if `range` isn't a list 
            range=[-float('inf'),range] # upper bound

        # (expected number of components) = (rest ewquivalent width)*(scaling factor)
        if ec:
            _ec=ec
        else:
            _ec=rew*sf # this may be replaced with other approaches

        if rzsv:
            zselected_ind, zselected_val=utils().range_filter(zscore,range,rsd=1)
            while len(zselected_ind)<_ec: # select the number of component base on `ec`
                range=[range[0],range[1]+increment]
                zselected_ind, zselected_val=utils().range_filter(zscore,range,rsd=1)
            return zselected_ind, zselected_val
        
        else:
            zselected_ind=utils().range_filter(zscore,range,rsd=0)
            while len(zselected_ind)<_ec: # select the number of component base on `ec`
                range=[range[0],range[1]+increment]
                zselected_ind=utils().range_filter(zscore,range,rsd=0)
            return zselected_ind

    def rem_fal_sig(self,signal,width,rfs=False): 
        r"""remove (false) signals that are narrower than `width` 

        Parameters
        ----------
        signal : list, 1D array
            list of indices of data points that considerd to be signals
        width : int
            how wide a true signal should bigger than
            the unit is number of data points
        rfs : bool, optional [default==False]
            whether to return false signal
            True: return false signal (`signal_fal`)
            False: no `signal_fal` returned

        Returns
        -------
        signal_rem_fal: list
            signal that remove false signals
        signal_fal: list
            list of false signals

        Examples
        -----
        If a 4th, 5th, 7th, 8th, 9th, and 10th signal points are considered signals,
        then 8th and 9th signal points will be kept, and the rest will be discarded.
        """
        signal_rem_fal=np.array([],dtype=int)

        if rfs: # if rfs==True, return false signals
            signal_fal=np.array([],dtype=int)
            for i in signal:
                neighbor=set(np.arange(i-width,i+width+1,1,dtype=int).tolist())
                if neighbor.issubset(signal):
                    signal_rem_fal=np.append(signal_rem_fal,i)
                else:
                    signal_fal=np.append(signal_fal,i)
            return signal_rem_fal, signal_fal
        
        else: # if rfs==False
            for i in signal:
                neighbor=set(np.arange(i-width,i+width+1,1,dtype=int).tolist())
                if neighbor.issubset(signal):
                    signal_rem_fal=np.append(signal_rem_fal,i)
            return signal_rem_fal

    # def rem_fal_sig2(self,signal,width,rfs):
        # remove false signal by loosing criteria
        # if rfs: # if rfs==True, return false signals
            # def _criteria(self,signal,ind,criteria):
            #     if signal[ind]

            # signal_rem_fal=[]
            # signal_fal=[]
            # for i in signal:
            #     neighbor=set(np.arange(i-width,i+width+1,1,dtype=int).tolist())
            #     if neighbor.issubset(signal):
            #         signal_rem_fal.append(i)
            #     else:
            #         signal_fal.append(i)
            # for j in signal_fal:
            #     if j
            # return signal_rem_fal, signal_fal

    def adaptive_median_filter(self, masked_flux, wlen):
        """adaptive median filter

        Parameters
        ----------
        masked_flux : numpy.ma.MaskedArray
            flux contains mask
        wlen : int
            length of the window

        Returns
        -------
        median_flux : 1D array
            flux after adaptive median filter
        """
        length = len(masked_flux)
        ind_array_masked = np.ma.array(
            np.arange(0, length), mask=masked_flux.mask, dtype=float
        )
        median_flux = np.array([], dtype=float)
        for i in range(length):
            [left_ind_array_masked, right_ind_array_masked] = np.split(
                ind_array_masked, [i]
            )

            liamc = left_ind_array_masked.compressed()
            riamc = right_ind_array_masked.compressed()

            if len(liamc) == 0:
                left_b = 0
            else:
                left_b = int(liamc[np.clip(-wlen, -len(liamc), 0)])

            if len(riamc) == 0:
                right_b = length
            else:
                right_b = int(riamc[np.clip(wlen, 0, len(riamc) - 1)])

            # [left_b,right_b]=spectool().rem_inval_epd_bnd(masked_flux.mask,i,wlen)

            median_flux = np.append(
                median_flux, np.median(masked_flux[left_b:right_b].compressed())
            )
        return median_flux

    def makesure_mask(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract the arguments, considering that `mask` may not be provided
            if len(args) == 2:  # If only two arguments are given, assume mask is None
                flux, threshold = args
                mask = None
            elif len(args) == 3:  # If all three arguments are given, unpack them
                flux, threshold, mask = args
            else:
                raise ValueError("Incorrect number of arguments.")

            if isinstance(flux, np.ma.MaskedArray):
                if mask is not None:
                    _mask = np.ma.mask_or(flux.mask, mask.astype(bool))
                else:
                    _mask = flux.mask
            else:
                if mask is not None:
                    _mask = mask
                else:
                    _mask = False

            _flux = np.ma.array(flux.data, mask=_mask)

            # [threshold[0], threshold[1]]-->[|threshold[0]|, |threshold[1]|]
            # if threshold[0] > 0:
            #     _threshold = [-threshold[0], threshold[1]]
            # elif threshold[1] < 0:
            #     _threshold=[]
            # else:
            #     _threshold = threshold

            _threshold=[-abs(threshold[0]),abs(threshold[1])]

            return func(self, _flux, _threshold, _mask)
        return wrapper

    # mask the outlier of the flux
    @makesure_mask
    def mask_outlier(self, flux, threshold, mask=None):
        """mask the outlier in the flux

        Parameters
        ----------
        flux : 1D array, numpy.ma.MaskedArray   \n
            flux to remove outlier, \n
            the flux can be either a usual `numpy.ndarray`  \n
            or a `numpy.ma.MaskedArray` containing mask,    \n
            the mask will be preserved and get union with the outliers.  \n

        threshold : list
            threshold of valid zscore range,  \n  
            [lower threshold, upper threshold], \n
            the valid range is computed as [mode-|lower threshold|, mode+|upper threshold|],    \n
            the data that have zscores out of the range are considered outliers.    \n

        mask : 1D array [default==None]
            addition mask create by `numpy.ma` module,  \n
            if `flux` is `MaskedArray`, \n 
            `mask` will get union with `flux.mask`  \n
            True: invalid   \n
            False: valid    \n
            the same size as `flux` \n
            >>> array([False, False, False, ..., False, False, False])
            

        Returns
        -------
        Success \n
        result : dict   \n
            * flux_masked : numpy.ma.MaskedArray  \n
                the flux containing mask,   \n
                the mask is the union of the original mask and the outliers.    \n

            * flux_nan : 1D array \n
                the data of `flux_masked` but replace the invalid data points with `NaN`    \n

            * zscore_flux_masked : numpy.ma.MaskedArray   \n
                the zscore containing mask, \n
                the mask is the same as `flux_masked.mask`  \n

            * zscore_flux_nan : 1D array  \n
                the data of `zscore_masked` but replace the invalid data points with `NaN`  \n

            * mask : 1D array \n
                the union of the original mask and the outliers \n
                >>> array([False, False, False, ..., False, False, False])  \n

            * outlier_ind : 1D array  \n
                the indices of the outliers,    \n
                the data points where `mask==True`  \n

            * mode : float    \n
                the true mode of zscore \n

            * range : list    \n
                valid range in zscore,    \n
                [lower bound, upper bound]
                =[true_mode(zscore) - |threshold[0]|, trude_mode(zscore) + |threshold[1]|]   \n

        Fail    \n
        res_debug : dict \n
            debugger

            * zscore_flux : list [1D arrays] \n
                whole record of zscores of flux  \n
            
            * mode : list [float]  \n
                whole record of modes   \n

            * zscore_flux_masked : list [1D MaskedArray]  \n
                whole record of zscore_flux \n

            * iteration : list [int]   \n
                nth iteration

        res_debug_zipped : list  \n
            zipped debugger

            list(   \n
                zip(    \n
                    res_debug['iteration'],  \n
                    res_debug['zscore_flux'],    \n
                    res_debug['mode'],   \n
                    res_debug['zscore_flux_masked']  \n
                )   \n
            )   \n

            the number of iteration in `res_debugger`   \n
            is mismatch with that in `res_debugger_zipped`  \n

        Warnings
        --------
        'All data points have been mask! Try to loose the threshold.'    \n

        Notes   \n
        -----
        flux can either be a raw flux or a normalized flux  \n

        """
        # --from @makesure_mask--#
        _flux, _threshold, _mask = flux, threshold, mask
        # --from @makesure_mask--#

        # zscore_flux = stats.zscore(_flux.filled(float("nan")), nan_policy="omit")
        # zscore_flux = utils().zscore(_flux)

        zscore_flux = utils().gzscore(_flux)

        mode = utils().true_mode(zscore_flux)
        # range
        # rg=[mode]

        zscore_flux_masked = np.ma.masked_outside(
            zscore_flux, mode + _threshold[0], mode + _threshold[1]
        )

        iteration=0

        res_debug={}
        res_debug['zscore_flux']=[zscore_flux]
        res_debug['#_NaN']=[sum(np.isnan(res_debug["zscore_flux"][iteration]))]
        res_debug['mode']=[mode]
        res_debug['zscore_flux_masked']=[zscore_flux]
        res_debug['iteration']=[iteration]

        while isinstance(zscore_flux_masked.mask, np.bool_) == False:
        # while ~isinstance(zscore_flux_masked.mask, np.bool_):
            iteration+=1
            res_debug['iteration'].append(iteration)

            # zscore_flux = stats.zscore(
            #     zscore_flux_masked.filled(float("nan")), nan_policy="omit"
            # )
            # zscore_flux = utils().zscore(zscore_flux_masked)

            zscore_flux = utils().gzscore(zscore_flux_masked)

            res_debug['zscore_flux'].append(zscore_flux)
            res_debug['#_NaN'].append(sum(np.isnan(res_debug["zscore_flux"][iteration])))

            if isinstance(zscore_flux_masked.mask,np.bool_):
                pass
            else:
                if sum(np.ma.masked_invalid(zscore_flux).mask)==len(_flux):
                    warnings.warn('All data points have been mask! Try to loose to threshold.')
                    return res_debug, res_debug_zipped
                    # raise st.StatisticsError('All data points have been mask!')

            # try:
            mode = utils().true_mode(zscore_flux)
            res_debug['mode'].append(mode)

            zscore_flux_masked = np.ma.masked_outside(
                zscore_flux, mode + _threshold[0], mode + _threshold[1]
            )
            res_debug['zscore_flux_masked'].append(zscore_flux_masked)

            # except st.StatisticsError:
            res_debug_zipped=list(
                                zip(
                                    res_debug["iteration"],
                                    res_debug['#_NaN'],
                                    res_debug["zscore_flux"],
                                    res_debug["mode"],
                                    res_debug["zscore_flux_masked"],
                                )
                            )                

        # --the same--#
        # _zscore_flux_masked = np.ma.masked_invalid(zscore_flux)
        _zscore_flux_masked = np.ma.masked_invalid(zscore_flux_masked)
        # --the same--#

        if isinstance(_mask, np.ndarray):
            _mask1 = np.ma.mask_or(_zscore_flux_masked.mask, _mask)
        else:
            _mask1 = _zscore_flux_masked.mask

        res = {}
        res["flux_masked"] = np.ma.array(flux, mask=_mask1)
        res["flux_nan"] = np.ma.array(flux, mask=_mask1, fill_value=float("nan")).filled()
        res["zscore_flux_masked"] = np.ma.array(_zscore_flux_masked, mask=_mask1)
        res["zscore_flux_nan"] = zscore_flux
        res["mask"] = _mask1
        res["outlier_ind"] = np.where(_mask1 == True)[0]
        res["mode"] = mode
        res["range"] = [mode + _threshold[0], mode + _threshold[1]]

        return res
    
    @makesure_mask
    def mask_absorption(self, flux, threshold, mask=None):
        """mask the absorption lins 

        Parameters
        ----------
        flux : 1D array, MaskedArray    \n
            the normalized flux that has passed `spectool().mask_outlier()` \n

        threshold : list    \n
            [lower threshold, upper threshold], \n
            the valid range is [mode-|lower threshold|, mode+|upper threshold|] \n

        mask : 1D array, list [default None]    \n
            additional mask array,  \n 
            if `flux` is `MaskedArray`  \n
            the final mask will be the union of `flux.mask` and `mask`  \n

        Returns
        -------
        result : dict   \n
            * flux_masked : numpy.ma.MaskedArray  \n
                the flux containing mask,   \n
                the mask is the union of the original mask and the outliers.    \n

            * flux_nan : 1D array \n
                the data of `flux_masked` but replace the invalid data points with `NaN`    \n

            * zscore_flux_masked : numpy.ma.MaskedArray   \n
                the zscore containing mask, \n
                the mask is the same as `flux_masked.mask`  \n

            * zscore_flux_nan : 1D array  \n
                the data of `zscore_masked` but replace the invalid data points with `NaN`  \n

            * mask : 1D array \n
                the union of the original mask and the outliers \n
                >>> array([False, False, False, ..., False, False, False])  \n

            * outlier_ind : 1D array  \n
                the indices of the outliers,    \n
                the data points where `mask==True`  \n

            * mode : float    \n
                the true mode of zscore \n

            * range : list    \n
                valid range in zscore,    \n
                [lower bound, upper bound]
                =[true_mode(zscore) - |threshold[0]|, trude_mode(zscore) + |threshold[1]|]   \n

        Notes
        -----
        The flux shold be normalized and removed outliers.
        """
        # calculate zscore
        zscore_flux_nan = stats.zscore(flux.filled(float("nan")), nan_policy="omit")
        zscore_flux_masked = np.ma.masked_invalid(zscore_flux_nan)

        # use zscore to calculate `mode`
        mode = utils().true_mode(zscore_flux_masked.filled(float("nan")))

        # mask invalid zscore
        _mask = np.ma.masked_outside(zscore_flux_masked, mode + threshold[0], mode + threshold[1]).mask

        # union of old mask and new mask
        _mask1=np.ma.mask_or(_mask,mask)

        _flux = np.ma.array(flux, mask=_mask1)

        res = {}
        res["flux_masked"] = _flux
        res["flux_nan"] = _flux.filled(float("nan"))
        res["zscore_flux_masked"] = zscore_flux_masked
        res["zscore_flux_nan"] = zscore_flux_nan
        res["mask"] = _mask1
        res["outlier_ind"] = np.where(_mask1 == True)[0]
        res["mode"] = mode
        res["range"] = [mode + threshold[0], mode + threshold[1]]

        return res
    

    def vcrop(self, spec: SpecTrum, line_wl: float, vspan: list, peak: bool|dict, ret_selected=False):
        """SpecTrum cropped by velocity span

        Parameters
        ----------
        spec : SpecTrum
            spec
        line_wl : float
            rest wavelength of transition
        vspan : list
            velocity span,
            calculate as [-| vspan[0] |,| vspan[1] |]
        peak : bool | dict


        Returns
        -------
        spec : SpecTrum
            `SpecTrum` that cropped by velocity span

        mask : np.ndarray[np.bool_]

        """
        xaxis_velocity_region = spec.xaxis_velocity_masked(line_wl).data
        _vspan = [-abs(vspan[0]), abs(vspan[1])]
        mask = (xaxis_velocity_region > _vspan[0]) & (xaxis_velocity_region < _vspan[1])

        flux_masked_region = spec.flux_masked[mask]
        xaxis_obs_region = spec.xaxis_obs[mask]
        err_region = spec.err[mask]
        z_ancestor = spec.z
        QSO_ID_ancestor = spec.QSO_ID
        dataset_ancestor = spec.dataset

        if len(spec.crys)==0:
            crys_ancestor=None
        else:
            crys_ancestor=np.array([],dtype=list)
            for i, ele in enumerate(spec.cry_headers):
                crys_ancestor=np.append(crys_ancestor,cry(spec.cry(i).mask[mask],ele))

        _spec=SpecTrum(
            flux_masked_region,
            err_region,
            z_ancestor,
            QSO_ID_ancestor,
            xaxis_obs_region,
            dataset=dataset_ancestor,
            crop=True,
            line_wl=line_wl,
            crys=crys_ancestor,
            peak=peak,
        )

        if ret_selected:
            return _spec, mask
        else:
            return _spec
    
    def ind2cry(self, spec : SpecTrum, ind : np.ndarray, header : str) -> np.ndarray[np.bool_]:
        z=np.zeros_like(spec.xaxis_obs)
        if len(ind)!=0:
        # try:
            z[ind]=1
        # except IndexError:
        #     return print(z,ind)
        else:
            pass
        
        return cry(z.astype(bool),header)
    
    # def ind2cry_new(self, spec : SpecTrum, ind : np.ndarray, header : str) -> np.ndarray[np.bool_]:
    #     z=np.ones_like(spec.xaxis_obs)
    #     z[ind]=0
    #     return cry(z.astype(bool),header)

    # def broadcast_cry_mask(self,spec,cry_header,mask,new_cry_header):

    def cry_and(self, cry_mask:np.ndarray, cry_mask1:np.ndarray)->np.ndarray[np.bool_]:
        """criteria AND logical operator

        `False` & `False` : `False`
        `False` & `True` : `False`
        `True` & `True` : `True`

        Parameters
        ----------
        cry_mask : np.ndarray
            first mask of `cry`
        cry_mask1 : np.ndarray
            second mask of `cry`

        Returns
        -------
        np.ndarray[np.ndarray_]
            result
        """
        return ~np.ma.mask_or(~cry_mask, ~cry_mask1)
    
    def cry_or(self, cry_mask:np.ndarray, cry_mask1:np.ndarray)->np.ndarray[np.bool_]:
        """criteria OR logical operator

        `False` & `False` : `False`
        `False` & `True` : `True`
        `True` & `True` : `True`

        Parameters
        ----------
        cry_mask : np.ndarray
            first mask of `cry`
        cry_mask1 : np.ndarray
            second mask of `cry`

        Returns
        -------
        np.ndarray[np.bool_]
            result
        """
        return np.ma.mask_or(cry_mask, cry_mask1)

    def find_centroid(self, xdata, ydata, i):
        a, b, _=utils().solve_quadratic(xdata[i-1 : i+2],ydata[i-1 : i+2])
        return -b/(2*a)
    
    def velocity_matching(
            self,
            vel: np.ndarray,
            range: tuple[np.ndarray, np.ndarray],
        )->tuple[np.ndarray, np.ndarray]:

        _range_left = range[0]
        _range_right = range[1]

        if not (len(_range_left) == len(_range_right)):
            raise ValueError(
                f"The size of `range_left` and `range_right` are different. The size of `range_left` is {len(_range_left)}, while th size of `range_right` is {len(_range_right)}."
            )

        mask_combined = False
        mask_rec = []
        for i, ele in enumerate(_range_left):
            mask = ~np.ma.masked_outside(vel, _range_left[i], _range_right[i]).mask

            if isinstance(mask, np.bool_):
                if mask == True:
                    _mask = np.ones_like(vel).astype(bool)
                elif mask == False:
                    _mask = np.zeros_like(vel).astype(bool)
            else:
                _mask = mask

            mask_combined = np.ma.mask_or(mask_combined, _mask)
            mask_rec.append(_mask)
        
        return mask_combined, np.sum(mask_rec,1).astype(bool)
            
    def get_velcom(
        self,
        spec: SpecTrum,
        cry_header: str,
        id: np.ndarray,
        peak_keys=None,
        new_peak_keys=None,
    ) -> list[VelCom]:
        """extract the `VelCom`s from `SpecTrum`

        Parameters
        ----------
        spec : SpecTrum
            the `SpecTrum` to extract `VelCom`
        cry_header : str
            the header of the cry that label the point of `VelCom`
        id : np.ndarray
            the id that specifies each 'VelCom'
        peak_keys : list[str], [default=None]
            peak_keys in SpecTrum,
            if `None`, all the keys in dict will be extracted
        new_peak_keys : list[str], [default=None]
            rename the peak_keys in VelCom,
            if `None`, all the keys in dict will be the same as in `spec.peak_dict`

        Returns
        -------
        list[VelCom]
            a list of `VelCom`
        """
        dt = {}
        if peak_keys is not None:
            for i, ele in enumerate(peak_keys):
                if new_peak_keys is None:
                    _ele = ele
                else:
                    _ele = new_peak_keys[i]

                if ele == "flux":
                    dt[_ele] = spec.flux_cry(cry_header).compressed()
                elif ele == "types":
                    pass
                else:
                    dt[_ele] = spec.peak_cry(ele, cry_header).compressed()

        else:
            for i, ele in enumerate(spec.peak_dict.keys()):
                dt[ele] = spec.peak_cry(ele, cry_header).compressed()
                dt["flux"] = spec.flux_cry(cry_header).compressed()

        keys = np.array(list(zip(dt.keys())))[:, 0]
        vals = np.array(list(zip(*dt.values())), dtype=list)
        vctype = spec.peak_cry("types", cry_header).compressed()
        res = np.array([], dtype=list)
        for i, ele in enumerate(vals):
            res = np.append(res, VelCom(keys, ele, vctype[i], id[i]))
        return res

    def linkage(
        self,
        vc: VelCom,
        vc1: VelCom,
        ingrediant: np.ndarray,
        w: list | np.ndarray,
        ret_detail=False,
    ):
        """calculate the linkage between 2 `VelCom`

        Parameters
        ----------
        vc : VelCom
            first `VelCom`
        vc1 : VelCom
            another `VelCom` to calculate their linkage
        ingrediant : np.ndarray
            ingrediant for linkage calculation
        w : list | np.ndarray
            weighting
        ret_detail : bool, optional
            whether to return detail, by default False

        Returns
        -------
        linkage
            linkage between 2 `VelCom`
        """
        igd = np.empty_like(ingrediant, dtype=float)
        igd1 = np.empty_like(ingrediant, dtype=float)
        for i, ele in enumerate(ingrediant):
            igd[i] = getattr(vc, ele)
            igd1[i] = getattr(vc1, ele)

        a = np.ma.masked_invalid(np.array(igd)).filled(0.0)
        a1 = np.ma.masked_invalid(np.array(igd1)).filled(0.0)

        d = a - a1

        # weighting
        _w = np.asarray(w)

        # linkage
        l = np.sqrt(
            np.average(
                np.square(d),
                weights=_w,
            ),
        )
        if ret_detail:
            return (l, a, a1, d, d * _w)
        else:
            return l
        
    def vc_binning(self, lt: list[VelCom], id: int):
        if len(lt) != 0:
            vel = np.empty_like(lt)
            flux = np.empty_like(lt)
            for i, ele in enumerate(lt):
                vel[i] = ele.vel
                flux[i] = ele.flux
            _vel = np.average(vel, weights=(1 - flux)**0.1)
            _flux=np.mean(flux)
        else:
            raise ValueError('What is this scenario?')
            _vel = lt[0].vel
            _flux = lt[0].flux

        return VelCom(
            np.array(["vel", "flux"]), np.array([_vel, _flux]), "velocity_binning", id
        )
    
    def calculate_logN(self,b: float, flux: float, wl: float, oscillation_strength: float):
        # from `VoigtFit.DataSet.interactive_components()`
        # y0 = max(y0/c_level, 0.01)
        # np.log10(-b * np.log(y0) / (1.4983e-15 * line.l0 * line.f))
        c_level=1.0 # the continuum level, assume normalized==1.0

        # flux must be within [0.01,0.99]
        _flux=max(flux/c_level, 1e-4)
        _flux1=min(_flux/c_level,1-1e-4)
        
        return np.log10(-b * np.log(_flux1) / (1.4983e-15 * wl * oscillation_strength))
# %%
class utils:
    """utilities
    """
    # def __init__(self):
    #     pass    

    def __init__(self) -> None:
        pass

    def orderpick(self,order_array,data):
        """ # Description
            pick the data from an array based on given order
            --------
            # Arguments
            order_array: 1D array
                order of data you want to extract
                
            data: 1D array
                whole data
            --------
            # Return
            the array of extracted data"""
        picked_data=np.zeros(0)
        for i in order_array:
            picked_data=np.append(picked_data,data[i])
        return picked_data
    
    def searchengine(self,Name,data,indices=True):
        r""" search string in an array

            Parameters
            --------
            Name: list or str
                list of name you want,            
                you can list as many as you want, 
                you can also use string if there is only one name, 
                don't need to write the full name 
            
            data: 1D array, dtype=str
                whole data to be searched, string in an array
            
            indices : bool [default=True]
                True : return indices of chosen element
                False : don't return indices of chosen element 

            Returns        
            --------
            Name_result: 1D array
                the full name of your search,     
                if there are similar name,        
                may return more than one result   
            
            Index_result: 1D array
                return the order of the search result, if indices=True

            Notes    
            ---------
            If you aren't sure about whether the strings contain capital,
            use `lazy_regex_search` instead.
            """
        if isinstance(Name,list): # if Name is a list
            # print('Name is a list')
            pass
        else: # Name isn't a list
            # print('Name isn\'t a list')
            Name=[Name]
            
        Name_result=np.zeros(0,dtype=list)
        for t in range(len(Name)):
            Name_result=np.append(Name_result,np.array([i for i in data if Name[t] in i]))

        if indices==True:
            Index_result=np.zeros(0,dtype=list)
            for u in range(len(Name_result)):
                Index_result=np.append(Index_result,np.where(data==Name_result[u]))

            return Name_result, Index_result
        
        if indices==False:
            return Name_result

    def lazy_regex_search(self,pattern,data,indices=True,ignorecase=True):
        """lazy regex search

        Parameters
        ----------
        pattern : str
            pattern to be searched
        data : list
            list of string
        indices : bool, optional [default==True]
            True: return Index_result
            False: no Index_result returned
        ignorecase : bool, optional [default==True]
            True: case insensitve
            False: case sensitive

        Returns
        -------
        Name_result: list
            list of string that match the `pattern`
        Index_result: list
            indices of above string in 'data', only returned when indices==True

        Notes
        -----
        If `pattern` contains "+",
        e.g. 'J115122+020426' please change to 'J115122\+020426'
        or use `utils().searchengine()`

        Examples
        --------
        >>> utils().lazy_regex_search(
            'zabs',spectool().abs_catalog('Mg')[1].columns.names
            )
        """
        if ignorecase==True:
            tosearch=re.compile(pattern,re.IGNORECASE)
        else:
            tosearch=re.compile(pattern)

        Name_result=[]

        if indices==True:
            Index_result=[]
            for i, col in enumerate(data):
                try: 
                    tosearch.search(col).group()
                    Name_result.append(col)
                    Index_result.append(i)
                except AttributeError:
                    pass
            return Name_result, Index_result
        # don't return Index_result
        else:
            for i, col in enumerate(data):
                try: 
                    tosearch.search(col).group()
                    Name_result.append(col)
                except AttributeError:
                    pass
            return Name_result

    #---can be replaced by `numpy.where((x<v1)|(x>v2))`---# 
    def range_filter(self,data,range,rsd=True):
        """ # Description
            find the value within given range
            --------
            # Arguments
            data: ndarray
                data to be filtered
            range: list
                [min,max]
            rsd: bool
                whether to return selected data
                True: return selected data (`data_selected`)
                False: no `data_selected` returned
            --------
            # Return 
            data_selected_indices: 1D array
                the array of the order of value within the range

            data_selected: 1D array
                the array of value within the range
            """
        #--try and check the following code in the future--#
        # (some criteria) & (some criteria)
        # e.g. (tp.diagonal_size(x)>5 ) & (tp.diagonal_size(x)<50)

        data_masked_data=np.ma.masked_outside(data,range[0],range[1])
        data_masked_nona_mask=np.ma.masked_invalid(data_masked_data).mask # mask `nan` and `inf`

        data_selected_indices=np.zeros(0,dtype=int)
        data_selected_indices=np.append(data_selected_indices,np.where(data_masked_nona_mask==False))

        if rsd:
            data_selected=utils().orderpick(data_selected_indices,data)
            return data_selected_indices, data_selected
        else:
            return data_selected_indices
    #---can be replaced by `numpy.where((x<v1)|(x>v2))`---# 

    # --- can be replaced by `numpy.argmax()`
    def find_global_max(self,xdata,ydata):
        """find the correspond x_value of max(ydata)

        Parameters
        ----------
        xdata : 1D array
            xdata
        ydata : 1D array
            xdata

        Returns
        -------
        x_val : float
            x value
        """
        y_val=np.amax(ydata)
        x_ind=np.where(ydata==y_val)
        x_val=xdata[x_ind[0]]
        return x_val

        # _xdata=np.asarray(xdata)
        # _ydata=np.asarray(ydata)
        # ind=int(_xdata[_ydata==max(_ydata)][0])
        # return _xdata[ind]

    def find_global_min(self,xdata,ydata):
        """find the correspond x_value of min(ydata)

        Parameters
        ----------
        xdata : 1D array
            xdata
        ydata : 1D array
            xdata

        Returns
        -------
        x_val : float
            x value
        """
        y_val=np.amin(ydata)
        x_ind=np.where(ydata==y_val)
        x_val=xdata[x_ind[0]]
        return x_val

    # def closest_values(self, be_search: np.ndarray, search: np.ndarray)->np.ndarray:
    #     """_summary_

    #     Parameters
    #     ----------
    #     be_search : np.ndarray
    #         _description_
    #     search : np.ndarray
    #         _description_

    #     Returns
    #     -------
    #     np.ndarray
    #         _description_
    #     """
    #     absolute_differences = np.abs(be_search[:, np.newaxis] - search)
    #     closest_indices = np.argmin(absolute_differences, axis=0)
    #     closest_values = be_search[closest_indices]
    #     return closest_values


    # def find_duplicate(self,data):
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             for t in range(data.shape[0]-i-1):
    #                 if data[i][j]==data[t+1][j]:

    def find_duplicate(self,data,indices=False):
        """find duplicates in array

        Parameters
        ----------
        data : 1D array
            array to find duplicate
        
        indices : bool, default to be False
            `True`: return indices of `dup` and `no_dup`
            `False`: no indices returned

        Returns
        -------
        dup : 1D array
            duplicate elements in `data`
        
        no_dup : 1D array 
            array with no duplicate elements

        dup_indices : 1D array
            return index of `data` of each element in `dup`, if `indices=True`

        no_dup_indices : 1D array
            return index of `data` of each element in `no_dup`, if `indices=True` 
        """

        if indices==False:
            no_dup=np.array([],dtype=list)
            dup=np.array([],dtype=list)
            for i in data:
                if i in no_dup:
                    dup=np.append(dup,i)
                else:
                    no_dup=np.append(no_dup,i)
            return dup, no_dup

        else:
            no_dup=np.array([],dtype=list)
            dup=np.array([],dtype=list)            
            no_dup_indices=np.array([],dtype=list)
            dup_indices=np.array([],dtype=list)

            for i in range(len(data)):
                if data[i] in no_dup:
                    dup=np.append(dup,data[i])
                    dup_indices=np.append(dup_indices,i)
                else:
                    no_dup=np.append(no_dup,data[i])
                    no_dup_indices=np.append(no_dup_indices,i)

            return dup, no_dup, dup_indices, no_dup_indices

    def compare_array(self, data1, data2):
        """compare the elements in 2 arrays

        Parameters
        ----------
        data1 : 1D array
            1st data to be compared
        data2 : 1D array
            2nd data to be compared

        Returns
        -------
        result : 1D array, dtype=bool
            True : this element is the same
            False : this element is different

        Raises
        ------
        IndexError
            data1 and data2 should have the same lens
        """
        if len(data1)!=len(data2): #check whether data1 and data2 have the same length
            raise IndexError("data1 has len=%s, but data2 has len=%s, they should have the same length"  %(len(data1),len(data2)))

        result=np.array([],dtype=bool)
        for i in range(len(data1)):
            if data1[i]==data2[i]:
                result=np.append(result,True)
            else:
                result=np.append(result,False)
        return result
    
    def check_elements_val(self,array,val):
        """check if the elements are all `val`
            e.g. 
            >>> lt=[None, None, None]
            >>> check_elements_val(lt,None)
            >>> True
        Parameters
        ----------
        array : array, list
            object to be checked
        val : float, None
            value of the element

        Returns
        -------
        bool
            whether all the element in the object is `val`
        """
        check=np.array([])
        for i in array:
            if i==val:
                check=np.append(check,True)
            else:
                check=np.append(check,False)

        if check.all():
            return True
        else:
            return False 

    def count_num_val(self,array,val):
        """count the number of element in an array that equal to `val`

        Parameters
        ----------
        array : 1D array
            array need to be counted
        val : int, float, bool
            value need to be counted

        Returns
        -------
        int
            number of element==val
        """
        # array==val will return a boolean array 
        # with `True` (1) if the element==val
        # with `False` (0) if the element!=val
        # `np.count_nonzero` will count the number of 1 in boolean array 
        num=np.count_nonzero(array==val)
        return num
    
    def prep_axvline(self,line_wl,line_list=None):
        axvline_list=[]
        if line_list is None:
            pass
        else:
            if isinstance(line_list,list):
                pass
            else:
                # special case for 1 line in line_list
                line_list=[line_list]

            for line in line_list:
                axvline=spectool().w2v(line_wl,line)
                axvline_list.append(axvline)
            return axvline_list

    #---don't work---#  
    def append_axvline(self,axvline_list,axvline):
        try:
            axvline_list.append(axvline)
        except NameError:
            axvline_list.append([])
    #---don't work---# 

    def Nones(self, dim, dtype):
        """create an array with all element=None and dimension=`dim`

        Parameters
        ----------
        dim : list
            dimension of the array
        dtype : data type
            data type of the array,
            e.g. object, list etc.

        Returns
        -------
        `dim` array
            array wiht all element=None and dimension=`dim`
        """
        arr=np.empty(dim,dtype)
        arr[:]=None
        return arr

    #--replaced by `scipy.stats.zscore`--#
    # def z_scores(self, data):
    #     z_scores = np.abs((data - np.mean(data)) / np.std(data))
    #     return z_scores
    #--replaced by `scipy.stats.zscore`--#

    # -- new version--#
    def rem_inval_epd_bnd(self,mask,center,wlen):
        r"""remove invalid data and expand boundary of the window, 
        compress the `MaskedArray`, and find the median in the array.

        Parameters
        ----------
        mask : 1D array
            array contain boolen which is generate by `numpy.ma` module,
            the array should be a complete array not a slice of original array.
            Notice that `True` means that the data point doesn't satisfy the criteria. 
        center : int
            index of the center of the window
        wlen : int
            half length of the window, 
        Returns
        -------
        boundary: list
            the modified boundary of the window, 
            [left boundary, right boundary]
        """

        length = len(mask)

        # build an array with value equal to index
        ind_array_masked = np.ma.array(
        np.arange(0, length), mask=mask, dtype=float
        )

        # split the array at the `center`
        [left_ind_array_masked, right_ind_array_masked] = np.split(
                ind_array_masked, [center]
            )
        
        # left subarray
        liamc = left_ind_array_masked.compressed()
        # right subarray
        riamc = right_ind_array_masked.compressed()

        if len(liamc) == 0:
            left_b = 0
        else:
            left_b = int(liamc[np.clip(-wlen, -len(liamc), 0)])

        if len(riamc) == 0:
            right_b = length
        else:
            # the -1 here is important for `np.clip(33000,0,33000)=33000`
            right_b = int(riamc[np.clip(wlen, 0, len(riamc) - 1)]) 

        return [left_b,right_b]
    # -- new version--#

    # --replaced with `mask_outliers()`--#
    def rem_outliers(self, threshold, data=None, zscore=None, wlen=5, ret_outlier=False, ret_zscore=False):
        # remove outliers and replace with the median of nearby flux
        if zscore is None:
            _zscores = stats.zscore(data,nan_policy='omit')
        else:
            _zscores=zscore

        if isinstance(threshold,list):
            outlier=np.where((_zscores<threshold[0]) | (_zscores>threshold[1]))[0]
        else:
            outlier=np.where(_zscores>threshold) # upper bound

        # mask the zscores that are out of range
        zscores_mask=np.ma.masked_outside(_zscores,threshold[0],threshold[1])
        # pass the mask to data
        data_mask=np.ma.array(data,mask=zscores_mask.mask)

        new_data=[]
        result={}

        if len(data_mask.compressed())==len(data_mask): # if there is no outlier
            new_data=data
            outlier=None
        else:
            # replace outliers with the median
            for ind, ele in enumerate(data):
                if ind in outlier:
                    bnd=utils().rem_inval_epd_bnd(data_mask.mask,ind,wlen)
                    new_data.append(np.median(data_mask[bnd[0]:bnd[1]].compressed()))
                else:
                    new_data.append(ele)
        new_zscores=stats.zscore(new_data,nan_policy='omit')
            
        result['new_data']=np.asarray(new_data)
        if ret_outlier:
            result['outlier_ind']=outlier
        if ret_zscore:
            result['new_zscore']=new_zscores
            
        return result
    # --replaced with `mask_outliers()`--#

    # --useless--#
    def get_middle_value(self,data):
        r"""return the value in the middle of the array

        Parameters
        ----------
        data : 1D array
            array

        Returns
        -------
        middle value: int, float
            middle value of `data`
        """
        l = len(data)
        if len(data) % 2 == 0:  # even
            return np.average(data[int(l / 2)] + data[int(l / 2) + 1])
        else:  # odd
            return data[int(l / 2) + 1]
    # --useless--#

    def test_ele_type(self,data,t):
        """test the type of the elements in the array

        Parameters
        ----------
        data : array
            array to be test
        t : type
            type to be test

        Raises
        ------
        TypeError
            if there is type that isn't `t`
        """

        # change the type into string
        s=str(t) # e.g. <class 'bool'>
        st=re.search(r'\'\w+\'',s).group() # e.g. 'bool'
        # makesure the `data` is list
        if isinstance(data,list):
            pass
        else:
            data=data.tolist()
        for i, ele in enumerate(data):
            if isinstance(ele,t):
                pass
            else:
                raise TypeError(f'Element {i} isn\'t {st}.')
        return True

    def true_mode(self, data, bins=1000, sigma=5,ret_detail=False):   
        """the mode after removing fake modes with gaussian filter 

        Parameters
        ----------
        data : 1D array, MaskedArray
            data (zscore) to calculated mode
        bins : int, optional [default=1000]
            seperate the value of `data` into how many bins 
        sigma : int, optional [default=5]
            the sigma of gaussian filter 
        ret_detail : bool, optional [default=False]
            whether to return the detail 

        Returns
        -------
        true_mode : float
            the true mode of `data`
        detail : dict
            keys: `counts`, `bins`, `bins_center`, `counts_gaussian`

        Example
        -------
        
        """

        # cannot contain NaN
        _data=np.ma.masked_invalid(data)
        counts, bins = np.histogram(_data.compressed(), bins)
        # the center value of the bins
        bins_center = (bins[1:]+bins[:-1])/2
        # gaussian filter to remove fake modes (too narrow)
        counts_gaussian = ni.gaussian_filter1d(counts,sigma)
        # find the x value of the peak location
        tmode = utils().find_global_max(bins_center,counts_gaussian)[0]

        # --debugger--#
        res={}
        res['counts']=counts
        res['bins']=bins
        res['bins_center']=bins_center
        res['counts_gaussian']=counts_gaussian
        # res['tmode']=tmode
        # return res
        # --debugger--#

        if ret_detail:
            return tmode, res
        else:
            return tmode

        # --old version--#
        """mode that don't account for starting point and ending point,
        because in KODIAQ spectra, there are a sequence of 0 flux in the starting and ending end,
        therefore, the mode might be 0 if the sequences are not excluded.

        Parameters
        ----------
        data : 1D array, MaskedArray
            flux or zscore

        Returns
        -------
        true_mode : float
            mode that don't account for starting point and ending point
        """

        # mask = np.ma.masked_equal(data, value=data[0]).mask
        # mask1 = np.ma.masked_equal(data, value=data[-1]).mask
        # mask2 = np.ma.masked_invalid(data).mask
        # _mask = np.ma.mask_or(mask, mask1)
        # _mask1 = np.ma.mask_or(_mask, mask2)
        # _data = np.ma.array(data, mask=_mask1)
        # return st.mode(_data.compressed())
        # --old version--#

    def solve_quadratic(self, xdata, ydata):
        X = np.array([xdata**2, xdata, np.ones_like(xdata)]).T
        return np.linalg.solve(X, ydata)
    
    def homogenize_diff(
            self, 
            data: np.ndarray, 
            order: int)->np.ndarray|tuple[np.ndarray,np.ndarray]:
        """ difference of the data
            `np.diff()` but with the same dimension,
            if the order of the diff is odd, there will be `left` and `right` diff

        Parameters
        ----------
        data : np.ndarray
            data to be diff
        order : int
            the order of the difference

        Returns
        -------
        np.ndarray|tuple[np.ndarray,np.ndarray]
            diff array

        Example
        -------
        data=[1,2,3,4,5]
        homogenize_diff(data,1)
        >>> (array([nan,  1.,  1.,  1.,  1.]),array([1.,  1.,  1.,  1., nan]))
        homogenize_diff(data,2)
        >>> array([nan,  0.,  0.,  0., nan])
        """
        
        if (order % 2) == 0:  # even
            diff = np.ones_like(data) * np.nan
            ind = int(order / 2)
            diff[ind:-ind] = np.diff(data, order)
            return diff
            # print('even')
        else:  # odd
            diff_left = np.ones_like(data) * np.nan
            diff_right = np.ones_like(data) * np.nan
            diff = np.diff(data, order)
            if order == 1:
                diff_left[order:] = diff
                diff_right[:-order] = diff
            else:
                ind_big = int((order + 1) / 2)
                ind_small = int((order - 1) / 2)
                diff_left[ind_big:-ind_small] = diff
                diff_right[ind_small:-ind_big] = diff
            return diff_left, diff_right
            # print('odd')

    def save_obj(self,obj:object,path:str):
        """save object

        Parameters
        ----------
        obj : object
            object to save, e.g. `SpecTrum`
        path : str
            path
        """
        saveObj = open(path,"wb")
        pickle.dump(obj, saveObj)
        saveObj.close()

    def load_obj(self,path:str) -> object:
        """load object

        Parameters
        ----------
        path : str
            path

        Returns
        -------
        object
            the loaded object
        """
        loadObj = open(path,"rb",)
        obj = pickle.load(loadObj)
        loadObj.close()

        return obj
    
    def zscore(self,data):
        if isinstance(data,np.ma.MaskedArray):
            return stats.zscore(data.filled(np.nan),nan_policy='omit')
        elif isinstance(data,np.ndarray):
            return stats.zscore(data,nan_policy='omit')
        else:
            raise UnhandleError(f'Only numpy.ndarray and np.ma.MaskedArray have been handled. {type(data)} has not been handled.')

    def gzscore(self,data):
        return stats.gzscore(np.ma.masked_less_equal(data,0.0).filled(np.nan),nan_policy='omit')

    def make_folder(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
    
# %%
#----!! This class has been aborted !!-----#
class multiaxes_graph:
    def __init__(self,whole_title='',rowxcol=[2,2],xdata=np.zeros((2,2,1)),ydata=np.zeros((2,2,1))):
        r""" Description
            !! This class has been aborted !!

            plot  multiaxes graph,               
            prepare xdata and ydata in advance(see Example below),
            and then put them in the argumments

            --------
            Arguments 

            whole_title : str
                title for the whole graph
            
            rowxcol: list [default=[2,2]]
                [row_num, col_num]
                    
            xdata: 3D array
                data of xaxis
                [row_num, col_num, xdata of each subgraph]
                    
            ydata: 3D array
                data of yaxis
                [row_num, col_num, ydata of each subgraph]

            -------
            Example:         
            >>> argm=(3,2,1)            
            >>> xdata=np.zeros(argm,dtype=list)    
            >>> ydata=np.zeros(argm,dtype=list) 
            >>> xdata[1,1,0]=np.arange(10)
            >>> ydata[1,1,0]=np.ones(11)
            """
        self._xdata=xdata
        self._ydata=ydata
        self._rowxcol=rowxcol
        self._whole_title=whole_title
        graph,axs=plt.subplots(self._rowxcol[0],self._rowxcol[1])
        graph.suptitle(whole_title)
        graph.tight_layout()
        for i in range(self._rowxcol[0]):
            for j in range(self._rowxcol[1]):
                axs[i,j].plot(self._xdata[i,j,0],self._ydata[i,j,0])
        # plt.show()

    @property
    def xdata(self):
        return self._xdata

    @property
    def ydata(self):
        return self._ydata

    @xdata.setter
    def xdata(self,val):
        self._xdata=val

    @ydata.setter
    def ydata(self,val):
        self._ydata=val

    def showplot(self):
        """show the plot
        """
        plt.show()

    def update(self):
        """update the plot
        """
        graph,axs=plt.subplots(self._rowxcol[0],self._rowxcol[1])
        graph.suptitle(self._whole_title)
        graph.tight_layout()
        for i in range(self._rowxcol[0]):
            for j in range(self._rowxcol[1]):
                axs[i,j].plot(self._xdata[i,j,0],self._ydata[i,j,0])   
        # plt.show()

    # def add_data(self,row_order,col_order):
    #     xdata=np.zeros(0)
#----!! This class has been aborted !!-----#
# %%
class twoaxes_graph:
    r""" # Description
        --------
        Subplot equal number of graph in each row, and more than one line in each graph.
        It will generate 1⨯1 graph by default.

        # Arguments
        --------
        whole_title : str
            title for the whole graph, 
            there will be nothing by default,
            if t

        rowXcolXlin : list [default=[1,1,1]]
            [num_row,num_col,max_num_lines],
            num_row : number of row
            num_col : number of column
            max_num_lines : maximum number of line (data)

        size : list [default=[6.4,4.8]]
            size for each figure, [width, height]

        axtitle :  list [default=None]
            title of each ax
            [pos_row,pos_col]
            pos_row : position of row
            pos_col : position of column

        xdata : 1D array [default=None]
            (pos_row,pos_col,ord_lin),
            pos_row : position of row
            pos_col : position of column
            ord_lin : order of line

        ydata : 1D array [default=None]
            (pos_row,pos_col,ord_lin),
            pos_row : position of row
            pos_col : position of column
            ord_lin : order of line  
            
        Example
        ------
        build the array in advanced 
        and put them into the arugment `xdata`, `ydata`
        >>> argm=(3,2,1)            
        >>> xdata=np.zeros(argm,dtype=list)    
        >>> ydata=np.zeros(argm,dtype=list) 
        >>> xdata[1,1,0]=np.arange(10)
        >>> ydata[1,1,0]=np.ones(10) 
        """
    def __init__(self,whole_title='',rowXcolXlin=None,size=[6.4, 4.8],
                line_name=None,axtitle=None,xdata=None,ydata=None):
        
        # title for whole figure
        self._whole_title=whole_title
        if rowXcolXlin==None:
            self._rowXcolXlin=[1,1,1]
        else:
            self._rowXcolXlin=rowXcolXlin

        # row, column and line 
        row=self._rowXcolXlin[0]
        col=self._rowXcolXlin[1]
        # lin=self._rowXcolXlin[2]

        # line_label
        if line_name is None:
            self._line_name=['']
        else:
            self._line_name=line_name

        # set up figure with `subplots()`
        self._fig, self._axes=plt.subplots(row,col,constrained_layout=1,
                                        figsize=(size[0]*col, size[1]*row))
        self._fig.suptitle(self._whole_title)

        # title for each ax
        if axtitle is None:
            self._axtitle=np.empty((self._rowXcolXlin[0],self._rowXcolXlin[1])
                                ,dtype=list)
        else:
            self._axtitle=axtitle

        # xdata for each ax
        if xdata is None:
            self._xdata=np.zeros(
                (self._rowXcolXlin[0],self._rowXcolXlin[1],self._rowXcolXlin[2]),
                dtype=list)
        else:
            self._xdata=xdata

        # ydata for each ax
        if ydata is None:
            self._ydata=np.zeros(
                (self._rowXcolXlin[0],self._rowXcolXlin[1],self._rowXcolXlin[2]),
                dtype=list)
        else:
            self._ydata=ydata
        
        # data for each ax
        self._data=np.zeros([self._rowXcolXlin[0],self._rowXcolXlin[1],self._rowXcolXlin[2],2],
                            dtype=list)
        for i in range(self._rowXcolXlin[0]):
            for j in range(self._rowXcolXlin[1]):
                for k in range(self._rowXcolXlin[2]):
                    self._data[i,j,k]=[self._xdata[i,j,k],self._ydata[i,j,k]]


    @property
    def xdata(self):
        for i in range(self._rowXcolXlin[0]):
            for j in range(self._rowXcolXlin[1]):
                for k in range(self._rowXcolXlin[2]):
                    self._xdata[i,j,k]=self._data[i,j,k,0]
        return self._xdata

    @property
    def ydata(self):
        for i in range(self._rowXcolXlin[0]):
            for j in range(self._rowXcolXlin[1]):
                for k in range(self._rowXcolXlin[2]):
                    self._ydata[i,j,k]=self._data[i,j,k,1]
        return self._ydata
    

#---------------------------#
# in confilict with `data`
    # @xdata.setter
    # def xdata(self,val):
    #     self._xdata=val

    # @ydata.setter
    # def ydata(self,val):
    #     self._ydata=val

    # @xdata.deleter
    # def xdata(self):
    #     del self._xdata
#-----------------------------#

    @property
    def data(self):          
        return self._data
    
    @data.setter
    def data(self,val):
        """data setter
        e.g.
        >>> twoaxes_graph.data[0,0,0]=[np.arange(10),np.arange(10)]
        Parameters
        ----------
        val : 2⨯3D array [default=None]
            [xdata, ydata]   
        """
        self._data=val

    @property
    def axtitle(self):
        return self._axtitle
    
    @axtitle.setter
    def axtitle(self,val):
        self._axtitle=val

    @property
    def ax_shape(self):
        return np.shape(self._axes)
    
    @property 
    def ax_size(self):
        return np.size(self._axes)

    @property
    def fig(self):
        return self._fig
#--------------------------#
    # def update_data(self):
    #     for i in range(self._rowXcolXlin[0]):
    #         for j in range(self._rowXcolXlin[1]):
    #             for k in range(self._rowXcolXlin[2]):
    #                 self._data[i,j,k]=[self._xdata[i,j,k],self._ydata[i,j,k]]
#--------------------------#   
    @property
    def ax(self):
        """axes of twoaxes_graph

        Returns
        -------
        list
            list of ax
        """
        return self._axes

    def plot_graph(self,ylim=None,show=True,label=True,legend=True,labelstr=[],axvline=None):
        """plot the graph

            Parameters
            --------
            ylim : list, optional, [default=None]
                set lower and upper bound of y-axis
                [lower bound, upper bound]
            show : bool, optional, [default=True]
                whether or not to show the graph with pop-up window
                recommand `False` if the graph is too large, or it will be distorted
            label : bool, optional, [default=True]
                whether or not to show the `xlabel` and `ylabel` of the each ax
                recommand `False` if each ax is too small
            legend : bool, optional, [default=True]
                whether or not to show the `legend` of each ax
                recommand `False` if each ax is too small
            labelstr : list, optional, [default=[]]
                [`xlabel`,`ylabel`]
                the string of xlabel and ylabel, only initiate when label is Ture
            axvline : list, optional, [default=None]
                only recommanded if there are multiple lines in an axis,
                otherwise, fig.ax.axvline() is more flexible
            
            Returns
            ------
            graph
            """
            #"""
            # size: list [default=[6.4, 4.8]]
            #     size for each figure, [width, height]

            # this_whole_title: str
            #     if you want to change the title of the graph,
            #     the following updated graph will inherit this title afterward.
            #"""


        # if line_name is None:
        #     pass
        # else:
        #     _line_name=line_name

        # I donn't need this block of code, 
        # I've set up fig & axes when I initiate the class
        #-----------------#
        # if this_whole_title=='':
        #     pass
        # else:
        #     self._whole_title=this_whole_title

        # row=self._rowXcolXlin[0]
        # col=self._rowXcolXlin[1]
        # # lin=self._rowXcolXlin[2]
                
        # fig, axes=plt.subplots(self._rowXcolXlin[0],self._rowXcolXlin[1], 
        #                              constrained_layout=1, 
        #                              figsize=(size[0]*col, size[1]*row))
        
        # fig.suptitle(self._whole_title)
        #-----------------#

        # I don't need this, this is in conflixt with `constrained_layout`
        # fig.tight_layout() 

        # if 1 row and 1 column
        if self._rowXcolXlin[0]==1 and self._rowXcolXlin[1]==1:
            for i in range(self._rowXcolXlin[2]):
                # self._axes.plot(self._xdata[0,0,i],self._ydata[0,0,i])
                self._axes.plot(self._data[0,0,i,0],self._data[0,0,i,1],alpha=0.5,label=self._line_name[i])
                self._axes.set_title(self._axtitle[0])

        # if 1 row and multiple columns
        elif self._rowXcolXlin[0]==1 and self._rowXcolXlin[1]!=1:
            for i in range(self._rowXcolXlin[1]):
                for j in range(self._rowXcolXlin[2]):
                    self._axes[i].plot(self._data[0,i,j,0],self._data[0,i,j,1],alpha=0.5,label=self._line_name[j])
                    self._axes[i].set_title(self._axtitle[0,i])

        # if multiple rows and 1 column
        elif self._rowXcolXlin[0]!=1 and self._rowXcolXlin[1]==1:
            for i in range(self._rowXcolXlin[0]):
                for j in range(self._rowXcolXlin[2]):
                    self._axes[i].plot(self._data[i,0,j,0],self._data[i,0,j,1],alpha=0.5,label=self._line_name[j])
                    self._axes[i].set_title(self.axtitle[i,0])

        # if multiple rows and multiple columns
        else:
            for i in range(self._rowXcolXlin[0]):
                for j in range(self._rowXcolXlin[1]):
                    for k in range(self._rowXcolXlin[2]):
                        self._axes[i,j].plot(self._data[i,j,k,0],self._data[i,j,k,1],alpha=0.5,label=self._line_name[k])
                        self._axes[i,j].set_title(self.axtitle[i,j])

        # whether to refine the graph 
        def _art(ax,ylim,label,legend,labelstr,axvline):
            _ax=ax

            # set ylim
            if ylim is None:
                pass
            else:
                _ax.set_ylim(ylim[0],ylim[1])

            # set label
            if label:
                if not labelstr:
                    # default
                    _ax.set_xlabel('Rel. Velocity (km/s)')
                    _ax.set_ylabel('Normalized Flux')
                else:
                    _ax.set_xlabel(labelstr[0])
                    _ax.set_ylabel(labelstr[1])
            else:
                pass

            # set legend
            if legend:
                _ax.legend()
            else:
                pass

            # set axvline
            if axvline is None:
                pass
            else:
                # default color cycler

                #--a good inspiration is in `trackpy.plots.fits()`; however, `get_color()` is method of pandas-#
                # [f.set_color(d.get_color()) for d, f in zip(datalines, fitlines)]
                #--a good inspiration is in `trackpy.plots.fits()`; however, `get_color()` is method of pandas-#

                #--another good inspiration--#
                # ax.get_lines()[1].set_color(ax.get_lines()[0].get_color())
                # see https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
                #--another good inspiration--#

                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                # if np.size(axvline)
                for i in range(len(axvline)):
                    if axvline[i] == "NL":
                        pass
                    else:
                        for j in range(len(axvline[i])):
                            _ax.axvline(axvline[i][j],color=colors[i],lw=1)
                        # _ax.axvline(axvline[i][0],color=colors[i],lw=1)

        # implement `_art` above
        if np.size(self._axes) == 1: 
            # special case for only 1 subplot, 
            # becase it will raise 
            # `AttributeError:'AxesSubplot' object has no attribute 'flatten'` 
            _art(self._axes,ylim,label,legend,labelstr,axvline)
        else:
            for ax in self._axes.flatten():
                _art(ax,ylim,label,legend,labelstr,axvline)

        # I don't need this, I've updated to `save_graph`.
        #----#
        # if save is False:
        #     pass
        # else:
        #     plt.savefig(os.path.join(path+self._whole_title),title=self._whole_title)
        #----#

        if show:
            plt.show()
        else:
            pass

    def save_graph(self,path,file_name=None,dpi=100):
        """save the graph

        Parameters
        ----------
        path : str
            the path where the graph is saved
        file_name : str, optional
            name of file, by default the title of the figure
        dpi : int, optional
            The resolution in dots per inch , by default 100
        """
        if file_name is None:
            self._file_name=self._whole_title
        else:
            self._file_name=file_name

        if os.path.exists(os.path.join(path+self._file_name)):
            nm=self._file_name.split('.')[0] # file name
            ft=self._file_name.split('.')[1] # file type
            self._file_name=nm+datetime.now().strftime("-%y-%m-%d-%H-%M-%S")+'.'+ft # add the time after file name
            # the format of day is "23-05-12-12-57-20" (2023/05/12, 12:57:20)
            #--a good example of warining can be found in--#
            #--C:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\trackpy\feature.py:425--#
            warnings.warn(f'The name of file already exists,it has been modified to "{self._file_name}"')

        self._fig.savefig(os.path.join(path+self._file_name),dpi=dpi,title=self._file_name)
        print(f'!!The figure has been saved to {path+self._file_name}!!')

# %%
class myset:
    def __int__(self):
        pass
    
    def crossmatch(self,data1,data2):
        r"""crossmatch of 2 data with `set().intersection()`,
            but I recommand using `numpy.intersect1d()`

        Parameters
        ----------
        data1 : 1D array
            1st data to crossmatch
        data2 : 1D array
            2nd data to crossmatch

        Returns
        -------
        Intersection : 1D array
            intersection of data1 and data1
        """
        Intersection_list=list((set(data1).intersection(set(data2)))) #transfer set to list
        Intersection=np.asarray(Intersection_list) #transfer list to array
        return np.sort(Intersection)

    def venn2_plot(self, data1, data2, name1='', name2='',savefig=False,path=None):
        """plot the venn digram with 2 set

        Parameters
        ----------
        data1 : 1D array, list, set
            1st data to plot venn diagram
        data2 : 1D array, list, set
            2nd data to plot venn diagram
        name1 : str, optional
            name of 1st data, by default ''
        name2 : str, optional
            name of 2nd data, by default ''
        """
        if isinstance(data1,set) and isinstance(data2,set):
            pass
        else:
            set1=set(data1)
            set2=set(data2)

        venn.venn2(subsets=[set1,set2], set_labels=[name1, name2])
        if savefig:
            plt.savefig(path,dpi=500)
        plt.show()
        
        # venn.venn2(subsets = (, 5, 2), set_labels = ('Group A', 'Group B'))

class mypie:
    def __init__(self, title, figsize=plt.rcParams["figure.figsize"],font_size=plt.rcParams['font.size']):
        """my pie chart

        Parameters
        ----------
        title : string
            title of the fig
        figsize : list, optional, by default rcParams["figure.figsize"]
            [width, height] of the figure
        font_size : int, optional, by default rcParams['font.size']
            font size of the pie chart
        """
        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._title=title
        plt.rc('font', size=font_size)

    def plot(self, label, size, use_percent=False):
        """plot the pie chart
        >>> labels = 'XQR-30', 'XSHOOTER archive'
        >>> sizes = [30, 12]
        >>> mypie.plot(labels,sizes)

        Parameters
        ----------
        label : string
            label of each part
        size : list
            value of each part
        use_percent : str, optional, by default False
            use percentage or true value
            percentage: True
            true value: False
        """
        if isinstance(size, list):
            pass
        else:
            size=list(size)

        total = sum(size)
        self._ax.set_title(self._title)

        if use_percent: # return percentage
            return self._ax.pie(size, labels=label,autopct='%1.1f%%')
        else: # return actual value
            return self._ax.pie(size, labels=label,autopct=lambda p: '{:.0f}'.format(p * total / 100))

    def show(self):
        """show pie chart
        """
        plt.show()

    def save(self, file_name,path, dpi=100,transparent=True):
        r"""_summary_

        Parameters
        ----------
        file_name : str
            file name
        path : raw string
            path of file
        dpi : int, optional, by default 100
            dpi
        transparent : bool, optional, by default True
            whether the background is transparent
            True: transparent
            False: opaque

        Raises
        ------
        SyntaxError
            The end should end with "\\\\"
        """
        if path[-1]!="\\":
            raise SyntaxError(r'The end should end with "\\"')
        else: 
            pass
        self._fig.savefig(os.path.join(path+file_name),dpi=dpi,transparent=transparent)
        print(f'!!The figure has been saved to {path+self._title}!!')

# %% 
class mystats:
    def __init__(self) -> None:
        pass 
    def linregress(xdata, ydata,weight=None)->[float, float, float, float, float]:
    # ---- Don't need `self` for `md.mystats.linregress()` ----#
    # def linregress(self, xdata, ydata,weight=None)->[object, float, float, float]:
    # ---- Don't need `self` for `md.mystats.linregress()` ----#
        r"""linear regression and uncertainty

        Parameters
        ----------
        xdata : np.ndarray
            xdata
        ydata : np.ndarray
            ydata
        weight : np.ndarray
            weighting, use the reciprocal of the square of y-uncertainty

        Returns
        -------
        [float, float, float, float, float]
            slope, intercept, unc_slope, unc_intercept, r_square

        Examples
        -------
        Unweight: LinregressResult.slope, LinregressResult.intercept

        Weighted: Linregression().coef_[0], Linregression().intercept_
            
        Notes
        -------
        Unweighted: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

        Weighted: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Reference
        -------
        P.198~199 of 
        "An Introduction to Error Analysis: The Study of Uncertainties in Physical Measurements" 
        2nd Edition by John R. Taylor for more details.

        """
        if weight is None:
            res = stats.linregress(xdata, ydata)
            slope=res.slope
            intercept=res.intercept
            r_square=res.rvalue
            
            unc_ydata = np.sqrt(
                np.sum((ydata - (xdata * slope + intercept)) ** 2) / (len(ydata) - 2)
            )

            Delta = len(xdata) * np.sum(xdata**2) - (np.sum(xdata)) ** 2
            unc_slope = unc_ydata * np.sqrt(len(ydata) / Delta)
            unc_intercept = unc_ydata * np.sqrt(np.sum(xdata**2) / Delta)

        else:
            regr=linear_model.LinearRegression()
            regr.fit(xdata.reshape(-1,1),ydata,weight)

            slope=regr.coef_[0]
            intercept=regr.intercept_

            r_square=r2_score(ydata,xdata*slope+intercept)

            Delta=np.sum(weight)*np.sum(weight*(xdata**2))-(np.sum(weight*xdata))**2
            unc_slope=np.sqrt(np.sum(weight)/Delta)
            unc_intercept=np.sqrt(np.sum(weight*(xdata**2))/Delta)


        # if weight is None:
        #     # return res, unc_slope, unc_intercept, unc_ydata
        #     # return res, unc_slope, unc_intercept
        #     slope=_slope
        #     intercept=_intercept

        # else:
        #     return regr, unc_slope, unc_intercept

        # return f"slope:{slope} \n intercept:{intercept} \n unc_slope:{unc_slope} \n unc_intercept:{unc_intercept} \n r_square:{r_square}"
        return slope, intercept, unc_slope, unc_intercept, r_square


    
# from lib
# LinregressResult = _make_tuple_bunch('LinregressResult',
#                                     ['slope', 'intercept', 'rvalue',
#                                     'pvalue', 'stderr'],
#                                     extra_field_names=['intercept_stderr'])
# %%
class myVoigtFit:
    def __init__(self):
        pass

    # def prepare_wfez(self, len, zdata):
    #     wl=np.zeros(len,dtype=list)
    #     flux=np.zeros_like(wl)
    #     err=np.zeros_like(wl)
    #     z=np.zeros_like(wl)
    #     for i in range(len):
    #         wl[i],flux[i],err[i]=np.loadtxt(r'D:\University_Course\Project\summer_program_2022\KODIAQ_specfile\%s.spec' %QSO_Name_selected[i],unpack=True)
    #         z[i]=zdata[i]

# %%
if __name__=='__main__':
    # b=twoaxes_graph(rowXcolXlin=[3,4,2],whole_title='Fig 2')
    # b.xdata[2,3,0]=np.arange(4)
    # b.ydata[2,3,0]=np.ones(4)
    # b.xdata[2,3,1]=np.arange(6)
    # b.ydata[2,3,1]=np.ones(6)*2
    # b.plot_graph(this_whole_title='Fig 2.0')
    # b.xdata[1,2,0]=np.arange(2)
    # b.ydata[1,2,0]=np.ones(2)*3
    # b.plot_graph()
    # b.xdata[0,1,1]=np.arange(3)
    # b.ydata[0,1,1]=np.ones(3)*2
    # b.plot_graph()
    # c=twoaxes_graph(rowXcolXlin=[2,2,3])
    # c.xdata[0,0,0]=np.arange(4)
    # c.ydata[0,0,0]=np.ones(4)*3
    # c.xdata[0,0,1]=np.arange(8)
    # c.ydata[0,0,1]=np.ones(8)
    # c.xdata[0,0,2]=np.arange(2)
    # c.ydata[0,0,2]=np.ones(2)*2
    # c.plot_graph()
    # d=spectrum().KODIAQ_zpick([1548,1550,2796,2803])
    # print(d)

    #--23/04/20--#
    # spectool().wl_trans("MgII_2796")
    #--23/04/20--#

    #--23/04/21--#
    # fig=twoaxes_graph('Eine')
    # fig.data[0,0,0]=[np.arange(10)/10,np.arange(10)/10]
    # fig.plot_graph([-0.2,1.2],1,0,0)
    #--23/04/21--#

    #--23/05/01--#
    # labels = 'XQR-30', 'XSHOOTER archive'
    # sizes = [30, 12]
    # total = sum(sizes)
    # pie=mypie('E-XQR-30',figsize=[12,6],font_size=30)
    # pie.plot(labels,sizes)
    # pie.show()
    # pie.save('E-XQR-30 Lfs',r"D:\University_Course\Project\summer_program_2022\Presentation\\")
    #--23/05/01--#

    #--23/06/28--#
    a = np.arange(5, dtype=float)
    a[2] = np.NaN
    # a[3] = np.PINF
    # index,data=utils().range_filter(a,[float('nan'),1])
    b=utils().range_filter(a,[float('nan'),1])
    #--23/06/28--#
# %%
