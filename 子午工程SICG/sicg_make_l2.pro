;+
; NAME:
;   sicg_make_l2
;
; PURPOSE:
;   Generate SICG L2 products from L1B/SPEI five-wavelength FITS files.
;   Products include peak intensity map, Doppler map, and line-width map.
;
; CATEGORY:
;   SICG data processing
;
; CALLING SEQUENCE:
;   sicg_make_l2, $
;   l1b_dir='D:\L1B\2024-11-13\', $
;   outroot='D:\L2\2024-11-13\', $
;   wave=637.4, $
;   /write_fits, /despike
;
; INPUT KEYWORDS:
;   l1b_dir     Directory containing L1B/SPEI FITS files.
;   l1b_files   Optional string array of input L1B FITS files. If set, l1b_dir is ignored.
;   outroot     Output root directory. INT/DOP/WID subdirectories are created below this path.
;   wave        Line center in nm. Supported default values: 637.4 and 530.3.
;   x0, y0      Occulter center in pixel coordinates, used to define the
;               annular fitting region.
;               The solar center recorded in the FITS header has not been updated.
;               Under good weather conditions, the occulter center is relatively stable.
;               If not provided, the default value is 1022.0,1022.0.
;   rlimb       Occulter radius, or the inner radial boundary used for L2 fitting, in pixels.
;               The occulter size is fixed for the current data processing.
;               If not provided, the default value is 487.0.
;   rmax        Outer fitting range above the occulter boundary in pixels.
;               The fitting region is defined as:
;               rlimb <= r <= rlimb + rmax
;               If not provided, the default value is 200.0.
;   bandpass    Instrumental FWHM in Angstrom. Default: 1.0 for 637.4 nm, 0.67 for 530.3 nm.
;   dop_limit   Absolute Doppler limit used in correction. Default: 200 km/s.
;   poly_order  Polynomial order for Doppler column trend correction. Default: 5.
;
; OUTPUT KEYWORDS:
;   int_out     Last generated intensity map.
;   dop_out     Last generated corrected Doppler map.
;   wid_out     Last generated line-width map.
;
; OPTIONAL KEYWORDS:
;   /write_fits Write L2 FITS products.
;   /despike
;     Apply a simple spatial smoothing to the input data cube before the
;     three-point inversion.
;   smooth_size
;     Size of the smoothing kernel in pixels.
;     The default value is 3, corresponding to a 3x3 eight-neighbor
;     average filter.
;   /quiet      Reduce printed messages.
;
; NOTES:
;   This program assumes the input L1B/SPEI data cube has dimensions [x,y,wave]
;   and contains five wavelength samples. It uses the middle three samples for
;   the analytic three-point Gaussian solution.
; 
;   More detailed descriptions of the processing steps are provided in the
;   comments of the individual subroutines below.

FUNCTION sicg_l2_datestr_from_hdr, hdr, file
; PURPOSE:
;   Extract the observation time from the FITS header and convert it to
;   a compact date-time string used in SICG L2 file names.
;   
; CALLING SEQUENCE:
;   datestr = sicg_l2_datestr_from_hdr(hdr, file)
;
; INPUTS:
;   hdr
;     FITS header string array of the input L1B/SPEI file.
;
;   file
;     Full path or file name of the input L1B/SPEI FITS file.
;     This is used as a fallback when DATE-OBS is missing from the header.
;
; OUTPUT:
;   datestr
;     A 14-character date-time string in the format:
;
;       YYYYMMDDhhmmss
;
;     For example, DATE-OBS = '2024-11-13T01:01:00.000'
;     will be converted to:
;
;       20241113010100
;
; NOTES:
;   This routine is mainly used to construct output L2 file names such as:
;
;     OYULO_SICG01_PKIM_L2_STP_YYYYMMDDhhmmss_V01.00_6374.FITS
;
;   The function does not modify the input header or data.
  compile_opt idl2

  date_obs = ''
  catch, err
  IF err EQ 0 THEN date_obs = strtrim(fxpar(hdr, 'DATE-OBS'), 2)
  catch, /cancel

  IF strlen(date_obs) GE 19 THEN BEGIN
    yy = strmid(date_obs, 0, 4)
    mm = strmid(date_obs, 5, 2)
    dd = strmid(date_obs, 8, 2)
    hh = strmid(date_obs, 11, 2)
    mi = strmid(date_obs, 14, 2)
    ss = strmid(date_obs, 17, 2)
    RETURN, yy+mm+dd+hh+mi+ss
  ENDIF

  ; fallback: search a 14-digit datetime in filename
  base = file_basename(file)
  FOR i=0, strlen(base)-14 DO BEGIN
    tmp = strmid(base, i, 14)
    IF stregex(tmp, '^[0-9]{14}$', /boolean) THEN RETURN, tmp
  ENDFOR

  RETURN, '00000000000000'
END

FUNCTION sicg_l2_obs_time_from_hdr, hdr
; PURPOSE:
;   Read the observation time from the FITS header.
;
; CALLING SEQUENCE:
;   obs_time = sicg_l2_obs_time_from_hdr(hdr)
;
; INPUT:
;   hdr
;     FITS header string array of the input L1B/SPEI file.
;
; OUTPUT:
;   obs_time
;     Observation time read from the DATE-OBS keyword.
;     If DATE-OBS is missing or invalid, the routine returns:
;
;       'DATE-OBS missing'
  compile_opt idl2

  date_obs = ''
  catch, err
  IF err EQ 0 THEN date_obs = strtrim(fxpar(hdr, 'DATE-OBS'), 2)
  catch, /cancel

  IF strlen(date_obs) GT 0 THEN RETURN, date_obs ELSE RETURN, 'DATE-OBS missing'
END

FUNCTION sicg_l2_filetime
; PURPOSE:
;   Generate the current file creation time for the output L2 FITS header.
;
; CALLING SEQUENCE:
;   filetime = sicg_l2_filetime()
;
; INPUTS:
;   None.
;
; OUTPUT:
;   filetime
;     Current system time string in the format:
;
;       YYYY-MM-DDTHH:MM:SS
;
;     This value is used for the FILETIME and DATE keywords in the
;     output L2 FITS header.
  compile_opt idl2

  caldat, systime(/julian), month_t, day_t, year_t, hour_t, mint_t, sec_t
  filetime = strmid(string(year_t), 8, 4)+'-'+strmid(string(month_t+100), 10, 2)+'-'+ $
             strmid(string(day_t+100), 10, 2)+'T'+strmid(string(hour_t+100), 10, 2)+':'+ $
             strmid(string(mint_t+100), 10, 2)+':'+strmid(string(sec_t+100), 8, 2)
  RETURN, filetime
END

PRO sicg_l2_set_default_geometry, wave, x0, y0, rlimb, rmax, bandpass
; PURPOSE:
;   Set default geometric parameters and instrumental bandpass for SICG L2
;   data processing.
;
; CALLING SEQUENCE:
;   sicg_l2_set_default_geometry, wave, x0, y0, rlimb, rmax, bandpass
;
; INPUT:
;   wave
;     Line center in nm. Common values are:
;
;       530.3  for Fe XIV 530.3 nm
;       637.4  for Fe X 637.4 nm
;
; INPUT/OUTPUT:
;   x0, y0      Occulter center in pixel coordinates, used to define the
;               annular fitting region.
;               The solar center recorded in the FITS header has not been updated.
;               Under good weather conditions, the occulter center is relatively stable.
;               If not provided, the default value is 1022.0,1022.0.
;
;   rlimb
;     Occulter radius, or the inner radial boundary used for L2 fitting, in pixels.
;     The occulter size is fixed for the current data processing.
;     If not provided, the default value is 487.0.
;
;   rmax
;     Outer fitting range above the occulter boundary in pixels.
;     The fitting region is defined as:
;
;       rlimb <= r <= rlimb + rmax
;
;     If not provided, the default value is 200.0.
;
;   bandpass
;     Instrumental FWHM in Angstrom.
;     If not provided, the default value is:
;
;       0.67 Angstrom for Fe XIV 530.3 nm
;       1.00 Angstrom for Fe X 637.4 nm
;
; NOTES:
;   The default disk center is set to (1022.0, 1022.0), because the solar
;   center in the FITS header has not been updated and the observed center
;   is usually stable under good weather conditions.
;
;   The parameter rlimb is used here as the fixed occulter radius or inner
;   radial boundary, rather than a newly fitted solar limb radius.
;
;   For special observing conditions or poorly centered images, x0, y0, and
;   rlimb should be checked and supplied manually.
  compile_opt idl2

  IF n_elements(x0) EQ 0 THEN x0 = 1022.0
  IF n_elements(y0) EQ 0 THEN y0 = 1022.0
  IF n_elements(rlimb) EQ 0 THEN rlimb = 487.0
  IF n_elements(rmax) EQ 0 THEN rmax = 200.0

  IF n_elements(bandpass) EQ 0 THEN BEGIN
    IF abs(wave-530.3) LE 0.2 THEN bandpass = 0.67 ELSE bandpass = 1.00
  ENDIF
END

FUNCTION sicg_l2_wave_offsets, wave, wvl0_ang
; PURPOSE:
;   Return the wavelength offsets of the five SICG sampling positions relative to the adopted line center.
;
; CALLING SEQUENCE:
;   psam = sicg_l2_wave_offsets(wave, wvl0_ang)
;
; INPUT:
;   wave
;     Line center in nm. Common values are:
;
;       530.3  for Fe XIV 530.3 nm
;       637.4  for Fe X 637.4 nm
;
; OUTPUT:
;   psam
;     Five wavelength offsets relative to the adopted line center, in Angstrom.
;
;   wvl0_ang
;     Adopted rest wavelength in Angstrom.
;     The value is returned through the argument.
;
; METHOD:
;   For the Fe XIV 530.3 nm line, the adopted sampling wavelengths are:
;
;       5301.00, 5302.00, 5302.65, 5303.30, 5306.00 Angstrom
;
;   and the adopted rest wavelength is:
;
;       5302.65 Angstrom
;
;   For the Fe X 637.4 nm line, the adopted sampling wavelengths are:
;
;       6372.00, 6373.70, 6374.35, 6375.00, 6377.00 Angstrom
;
;   and the adopted rest wavelength is:
;
;       6374.35 Angstrom
;
;   The routine returns the wavelength offsets:
;
;       psam = sampling_wavelengths - wvl0_ang
;
; NOTES:
;   The L2 three-point Gaussian inversion uses the middle three samples,
;   with indices [1, 2, 3].
;
;   For 637.4 nm, these are:
;
;       6373.70, 6374.35, 6375.00 Angstrom
;
;   For 530.3 nm, these are:
;
;       5302.00, 5302.65, 5303.30 Angstrom
  compile_opt idl2

  IF abs(wave-530.3) LE 0.2 THEN BEGIN
    wvl0_ang = 5302.65
    RETURN, [5301.00, 5302.00, 5302.65, 5303.30, 5306.00] - wvl0_ang
  ENDIF ELSE BEGIN
    wvl0_ang = 6374.35
    RETURN, [6372.00, 6373.70, 6374.35, 6375.00, 6377.00] - wvl0_ang
  ENDELSE
END

PRO sicg_l2_make_annulus, nx, ny, x0, y0, rlimb, rmax, mask
; PURPOSE:
;   Create an annular mask for the coronal region used in SICG L2 fitting.
;
; CALLING SEQUENCE:
;   sicg_l2_make_annulus, nx, ny, x0, y0, rlimb, rmax, mask
;
; INPUTS:
;   nx
;     Image size in the x direction.
;
;   ny
;     Image size in the y direction.
;
;   x0
;     Occulter center x coordinate in pixels.
;
;   y0
;     Occulter center y coordinate in pixels.
;
;   rlimb
;     Occulter radius, or the inner radial boundary used for L2 fitting,
;     in pixels.
;
;   rmax
;     Outer fitting range above the occulter boundary in pixels.
;
; OUTPUT:
;   mask
;     Two-dimensional annular mask with dimensions [nx, ny].
;     Pixels satisfying
;
;       rlimb <= r <= rlimb + rmax
;
;     are set to 1. Other pixels are set to 0.
  compile_opt idl2

  x = rebin(reform(findgen(nx), nx, 1), nx, ny)
  y = rebin(reform(findgen(ny), 1, ny), nx, ny)
  rr = sqrt((x-x0)^2 + (y-y0)^2)
  mask = (rr GE rlimb) AND (rr LE (rlimb+rmax))
END

PRO sicg_l2_smooth, cube, mask, smooth_size=smooth_size
; PURPOSE:
;   Smooth the input SICG five-wavelength data cube using a simple
;   neighbor-average filter.
;
; CALLING SEQUENCE:
;   sicg_l2_smooth, cube, mask, smooth_size=smooth_size
;
; INPUT/OUTPUT:
;   cube
;     Input SICG data cube with dimensions [x, y, wave].
;     The cube is modified in place.
;
; INPUT:
;   mask
;     Two-dimensional annular mask defining the region to be processed.
;     Only pixels inside the mask are smoothed.
;
; OPTIONAL KEYWORD:
;   smooth_size
;     Size of the smoothing kernel in pixels.
;     If not provided, the default value is 3.
;     For smooth_size=3, the routine applies a 3x3 eight-neighbor
;     average filter. The central pixel is excluded from the average.

  compile_opt idl2

  sz = size(cube)
  IF sz[0] NE 3 THEN RETURN
  nw = sz[3]

  ; Set the default smoothing kernel size.
  IF n_elements(smooth_size) EQ 0 THEN smooth_size = 3L
  smooth_size = long(smooth_size)

  ; The kernel size should be an odd integer.
  IF (smooth_size LT 3) THEN smooth_size = 3L
  IF (smooth_size MOD 2) EQ 0 THEN smooth_size = smooth_size + 1L

  kernel_size = smooth_size

  kernel = fltarr(kernel_size, kernel_size) + 1.0
  kernel[kernel_size/2L, kernel_size/2L] = 0.0
  kernel = kernel / total(kernel, /preserve_type)

  FOR iw=0, nw-1 DO BEGIN
    d = reform(cube[*,*,iw])
    s = convol(d, kernel, /edge_zero)

    ind = where(mask AND finite(d) AND finite(s) AND (abs(s-d) GT 0.0), nind)
    IF nind GT 0 THEN d[ind] = s[ind]

    cube[*,*,iw] = d
  ENDFOR
END

PRO sicg_l2_three_point, cube, wave, bandpass, x0, y0, rlimb, rmax, inten, dop, wid
; PURPOSE:
;   Derive SICG L2 spectral parameters from a five-wavelength L1B/SPEI
;   data cube using an analytic three-point Gaussian method(Tian et al. 2013).
;   
; CALLING SEQUENCE:
;   sicg_l2_three_point, cube, wave, bandpass, x0, y0, rlimb, rmax, $
;                         inten, dop, wid
;
; INPUTS:
;   cube
;     Input L1B/SPEI data cube with dimensions [x, y, wave].
;     The cube is expected to contain five wavelength samples.
;
;   wave
;     Line center in nm. Common values are:
;
;       530.3  for Fe XIV 530.3 nm
;       637.4  for Fe X 637.4 nm
;
;   bandpass
;     Instrumental FWHM in Angstrom.
;     Typical values are:
;
;       0.67 Angstrom for Fe XIV 530.3 nm
;       1.00 Angstrom for Fe X 637.4 nm
;
;   x0
;     Occulter disk center x coordinate in pixels.
;
;   y0
;     Occulter disk center y coordinate in pixels.
;
;   rlimb
;     Occulter radius, or inner radial boundary used for fitting,
;     in pixels.
;
;   rmax
;     Outer fitting range above the occulter boundary in pixels.
;     The fitting region is:
;
;       rlimb <= r <= rlimb + rmax
;
; OUTPUTS:
;   inten
;     Peak intensity map derived from the Gaussian line profile.
;     Pixels outside the fitting region or with invalid fitting results
;     are set to NaN.
;
;   dop
;     Doppler velocity map in km/s.
;     The velocity is calculated from the fitted line-center shift:
;
;       v = delta_lambda / lambda_0 * c
;
;     where c = 3.0e5 km/s.
;
;   wid
;     Line-width map in Angstrom.
;     This is the FWHM after instrumental broadening is removed:
;
;       wid = sqrt(FWHM_obs^2 - FWHM_inst^2)
  compile_opt idl2

  sz = size(cube)
  nx = sz[1]
  ny = sz[2]
  nw = sz[3]

  inten = fltarr(nx, ny) + !values.f_nan
  dop   = fltarr(nx, ny) + !values.f_nan
  wid   = fltarr(nx, ny) + !values.f_nan

  IF nw LT 5 THEN BEGIN
    message, 'Input cube must contain five wavelength samples.', /continue
    RETURN
  ENDIF

  psam = sicg_l2_wave_offsets(wave, wvl0_ang)
  vsam = [1, 2, 3]
  dw = abs(psam[vsam[2]] - psam[vsam[1]])
  gf = bandpass / (2.0 * sqrt(2.0 * alog(2.0)))

  sicg_l2_make_annulus, nx, ny, x0, y0, rlimb, rmax, mask

  good_width_count = 0L
  bad_count = 0L

  FOR ix=0, nx-1 DO BEGIN
    FOR iy=0, ny-1 DO BEGIN
      IF mask[ix, iy] THEN BEGIN
        isam = reform(cube[ix, iy, *])
        IF min(isam[vsam]) GT 0.0 THEN BEGIN
          a = alog(isam[vsam[2]] / isam[vsam[1]])
          b = alog(isam[vsam[0]] / isam[vsam[1]])

          IF finite(a) AND finite(b) AND ((a+b) LT 0.0) THEN BEGIN
            wo = sqrt(-2.0/(a+b)) * dw
            wfwhm = wo * (2.0*sqrt(2.0*alog(2.0))) / sqrt(2.0)
            lam0 = wo^2 / 4.0 / dw * (a-b) + psam[vsam[1]]
            vel = lam0 / wvl0_ang * 3.0e5

            IF wfwhm GT bandpass THEN BEGIN
              gsfwhm_c = sqrt(wfwhm^2 - bandpass^2)
              gs_c = gsfwhm_c / (2.0*sqrt(2.0*alog(2.0)))
              gso = wo / sqrt(2.0)

              denom = sqrt(2.0*!pi) * gs_c * gf / gso * exp(-0.5*((lam0-psam[2])^2/gso^2))
              IF denom GT 0.0 THEN BEGIN
                inten[ix,iy] = isam[2] / denom
                dop[ix,iy] = vel
                wid[ix,iy] = gsfwhm_c
                good_width_count = good_width_count + 1L
              ENDIF ELSE bad_count = bad_count + 1L
            ENDIF ELSE bad_count = bad_count + 1L
          ENDIF ELSE bad_count = bad_count + 1L
        ENDIF ELSE bad_count = bad_count + 1L
      ENDIF
    ENDFOR
  ENDFOR

  print, 'Valid fitted pixels: ', good_width_count
  print, 'Rejected fitted pixels: ', bad_count
END

PRO sicg_l2_correct_doppler, dop, x0, y0, rlimb, rmax, dop_corr, dop_col_median, dop_limit=dop_limit, poly_order=poly_order
; PURPOSE:
;   Correct the SICG Doppler map by subtracting the column-wise median
;   background velocity.
;
; CATEGORY:
;   SICG L2 data processing utility
;
; CALLING SEQUENCE:
;   sicg_l2_correct_doppler, dop, x0, y0, rlimb, rmax, $
;                           dop_corr, dop_col_median, $
;                           dop_limit=dop_limit, poly_order=poly_order
;
; INPUTS:
;   dop
;     Input Doppler velocity map in km/s.
;
;   x0
;     Occulter disk center x coordinate in pixels.
;
;   y0
;     Occulter disk center y coordinate in pixels.
;
;   rlimb
;     Occulter radius, or inner radial boundary used for correction,
;     in pixels.
;
;   rmax
;     Outer correction range above the occulter boundary in pixels.
;     The correction region is:
;
;       rlimb <= r <= rlimb + rmax
;
; OUTPUTS:
;   dop_corr
;     Corrected Doppler velocity map in km/s.
;     In the current version, each x column is corrected by subtracting
;     its own median Doppler value:
;
;       dop_corr[ix,*] = dop[ix,*] - dop_col_median[ix]
;
;   dop_col_median
;     One-dimensional array containing the median Doppler velocity of
;     each x column.
;
; OPTIONAL KEYWORDS:
;   dop_limit
;     Absolute Doppler velocity limit used when calculating the column
;     median. Only pixels satisfying
;
;       abs(dop) < dop_limit
;
;     are used. The default value is 200 km/s.
;
;   poly_order
;     Polynomial order used by the original polynomial-fit correction.
;     This keyword is kept for compatibility and future use.
;     In the current median-subtraction version, it is not used.
;     The default value is 5.
  compile_opt idl2
  
  IF n_elements(dop_limit) EQ 0 THEN dop_limit = 200.0
  IF n_elements(poly_order) EQ 0 THEN poly_order = 5

  sz = size(dop)
  nx = sz[1]
  ny = sz[2]

  sicg_l2_make_annulus, nx, ny, x0, y0, rlimb, rmax, mask

  dop_col_median = fltarr(nx) + !values.f_nan

  ; Calculate the median Doppler shift for each x column.
  FOR ix=0, nx-1 DO BEGIN
    col = reform(dop[ix,*])
    ind = where(mask[ix,*] AND finite(col) AND (abs(col) LT dop_limit), cnt)
    IF cnt GT 5 THEN dop_col_median[ix] = median(col[ind])
  ENDFOR

  ; Direct median subtraction.
  dop_corr = dop
  FOR ix=0, nx-1 DO BEGIN
    IF finite(dop_col_median[ix]) THEN BEGIN
      dop_corr[ix,*] = dop[ix,*] - dop_col_median[ix]
    ENDIF ELSE BEGIN
      dop_corr[ix,*] = dop[ix,*]
    ENDELSE
  ENDFOR

  ; -------------------------------------------------------------------------
  ; Original polynomial-fit correction. Kept here for future use.
  ; Uncomment this block if you want to subtract the fitted smooth trend
  ; instead of the direct column-wise median.
  ;
  ; xx = findgen(nx)
  ; valid = where(finite(dop_col_median), nvalid)
  ;
  ; dop_corr = dop
  ; IF nvalid GT poly_order+2 THEN BEGIN
  ;   coeff = poly_fit(xx[valid], dop_col_median[valid], poly_order)
  ;   trend = poly(xx, coeff)
  ;   FOR ix=0, nx-1 DO dop_corr[ix,*] = dop[ix,*] - trend[ix]
  ; ENDIF ELSE BEGIN
  ;   message, 'Too few valid columns for polynomial Doppler correction. Return raw Doppler map.', /continue
  ; ENDELSE
  ; -------------------------------------------------------------------------

END

PRO sicg_l2_update_hdr, hdr, content, quantity, filetime
; PURPOSE:
;   Update the FITS header keywords for SICG L2 data products.
;
; CALLING SEQUENCE:
;   sicg_l2_update_hdr, hdr, content, quantity, filetime
;
; INPUT/OUTPUT:
;   hdr
;     FITS header string array.
;     The header is updated in place.
;
; INPUTS:
;   content
;     Description of the L2 data product, such as:
;
;       'Coronal Peak-Intensity Maps'
;       'Coronal Dopplergrams'
;       'Coronal Line-Width Maps'
;
;   quantity
;     Physical quantity or unit of the output data, such as:
;
;       '1.E-06 B/Bsun'
;       'km/s'
;       'Angstrom'
;
;   filetime
;     File creation time string, usually generated by sicg_l2_filetime.
;     The recommended format is:
;
;       YYYY-MM-DDTHH:MM:SS
  compile_opt idl2

  fxaddpar, hdr, 'DATA_LEV', 'L2'
  fxaddpar, hdr, 'CONTENT', content
  fxaddpar, hdr, 'QUANTITY', quantity
  fxaddpar, hdr, 'FILETIME', filetime
  fxaddpar, hdr, 'DATE', filetime, 'time when this file was initially created'
END

PRO sicg_l2_write_one, outfile, map, hdr0, content, quantity
; PURPOSE:
;   Write one SICG L2 data product to a FITS file.
;
; CATEGORY:
;   SICG L2 data processing utility
;
; CALLING SEQUENCE:
;   sicg_l2_write_one, outfile, map, hdr0, content, quantity
;
; INPUTS:
;   outfile
;     Full output FITS file name, including the directory path.
;
;   map
;     Two-dimensional L2 data map to be written.
;     Examples include:
;
;       intensity map
;       Doppler velocity map
;       line-width map
;
;   hdr0
;     Original FITS header from the input L1B/SPEI file.
;     This header is copied and then updated for the L2 output.
;
;   content
;     Description of the L2 data product, such as:
;
;       'Coronal Peak-Intensity Maps'
;       'Coronal Dopplergrams'
;       'Coronal Line-Width Maps'
;
;   quantity
;     Physical quantity or unit of the output data, such as:
;
;       '1.E-06 B/Bsun'
;       'km/s'
;       'Angstrom'
  compile_opt idl2

  hdr = hdr0
  filetime = sicg_l2_filetime()
  sicg_l2_update_hdr, hdr, content, quantity, filetime
  writefits, outfile, float(map), hdr
END

PRO sicg_make_l2, l1b_dir=l1b_dir, l1b_files=l1b_files, outroot=outroot, $
                  wave=wave, x0=x0, y0=y0, rlimb=rlimb, rmax=rmax, $
                  bandpass=bandpass, dop_limit=dop_limit, poly_order=poly_order, $
                  smooth_size=smooth_size, $
                  int_out=int_out, dop_out=dop_out, wid_out=wid_out, $
                  write_fits=write_fits, despike=despike, quiet=quiet
; PURPOSE:
;   Generate SICG L2 products from L1B/SPEI five-wavelength FITS files.
;
;   The output L2 products include:
;
;     1. Intensity map
;     2. Doppler velocity map
;     3. Line-width map
;
; CALLING SEQUENCE:
;   sicg_make_l2, $
;     l1b_dir='D:\L1B\2024-11-13\', $
;     outroot='D:\L2\2024-11-13\', $
;     wave=637.4, $
;     x0=1022.0, y0=1022.0, $
;     rlimb=487.0, rmax=200.0, $
;     /write_fits
;
; INPUT KEYWORDS:
;   l1b_dir
;     Directory containing the input L1B/SPEI FITS files.
;     If l1b_files is not provided, the program searches this directory
;     for files matching the selected wavelength.
;
;   l1b_files
;     Optional string array of input L1B/SPEI FITS files.
;     If this keyword is provided, l1b_dir is ignored.
;
;   outroot
;     Output root directory for the L2 products.
;     Three subdirectories are created under this path:
;
;       INT
;       DOP
;       WID
;
;   wave
;     Line center in nm. Common values are:
;
;       530.3  for Fe XIV 530.3 nm
;       637.4  for Fe X 637.4 nm
;
;     If not provided, the default value is 637.4.
;
;   x0, y0
;     Occulter disk center in pixel coordinates.
;     If not provided, default values are set by sicg_l2_set_default_geometry.
;
;   rlimb
;     Occulter radius, or inner radial boundary used for L2 fitting,
;     in pixels.
;
;   rmax
;     Outer fitting range above the occulter boundary in pixels.
;     The fitting region is defined as:
;
;       rlimb <= r <= rlimb + rmax
;
;   bandpass
;     Instrumental FWHM in Angstrom.
;     If not provided, the default value is:
;
;       0.67 Angstrom for Fe XIV 530.3 nm
;       1.00 Angstrom for Fe X 637.4 nm
;
;   dop_limit
;     Absolute Doppler velocity limit used in the column-wise median
;     correction. Only pixels with:
;
;       abs(dop) < dop_limit
;
;     are used to estimate the column median.
;     The default value is 200 km/s.
;
;   poly_order
;     Polynomial order used by the previous polynomial trend correction.
;     This keyword is kept for compatibility and future use.
;     In the current version, the Doppler correction uses direct
;     column-wise median subtraction.
;
; OUTPUT KEYWORDS:
;   int_out
;     Last generated peak intensity map.
;
;   dop_out
;     Last generated corrected Doppler velocity map.
;
;   wid_out
;     Last generated line-width map.
;
; OPTIONAL KEYWORDS:
;   /write_fits
;     Write the generated L2 products to FITS files.
;     If this keyword is not set, the program only returns the output maps
;     through int_out, dop_out, and wid_out.
;
;   /despike
;     Apply simple spatial smoothing to the input data cube before the
;     three-point inversion.
;     In this program, /despike calls sicg_l2_smooth.
;
;   /quiet
;     Reduce printed messages during processing.
  compile_opt idl2

  IF n_elements(wave) EQ 0 THEN wave = 637.4
  sicg_l2_set_default_geometry, wave, x0, y0, rlimb, rmax, bandpass

  os = !version.os
  IF os EQ 'Win32' THEN slash = '\' ELSE slash = '/'

  IF n_elements(outroot) EQ 0 THEN outroot = '.'+slash+'L2'+slash
  int_dir = outroot + slash + 'INT' + slash
  dop_dir = outroot + slash + 'DOP' + slash
  wid_dir = outroot + slash + 'WID' + slash

  IF n_elements(l1b_files) EQ 0 THEN BEGIN
    IF n_elements(l1b_dir) EQ 0 THEN BEGIN
      message, 'Please set l1b_dir or l1b_files.', /continue
      RETURN
    ENDIF

    IF abs(wave-530.3) LE 0.2 THEN expo = '5303' ELSE expo = '6374'
    l1b_files = file_search(l1b_dir + slash + '*SPEI_L1B*_'+expo+'.FITS', count=nfiles)
    IF nfiles EQ 0 THEN l1b_files = file_search(l1b_dir + slash + '*L1B*_'+expo+'.FITS', count=nfiles)
  ENDIF ELSE BEGIN
    nfiles = n_elements(l1b_files)
    IF abs(wave-530.3) LE 0.2 THEN expo = '5303' ELSE expo = '6374'
  ENDELSE

  IF nfiles EQ 0 THEN BEGIN
    message, 'No L1B FITS files found.', /continue
    RETURN
  ENDIF

  l1b_files = l1b_files[sort(l1b_files)]
  IF NOT keyword_set(quiet) THEN print, 'Number of L1B files: ', nfiles

  IF keyword_set(write_fits) THEN BEGIN
    file_mkdir, int_dir
    file_mkdir, dop_dir
    file_mkdir, wid_dir
  ENDIF

  FOR ifile=0, nfiles-1 DO BEGIN
    file = l1b_files[ifile]
    IF NOT keyword_set(quiet) THEN print, 'Processing: ', file

    cube = readfits(file, hdr)
    cube = float(cube)

    sz = size(cube)
    IF sz[0] NE 3 THEN BEGIN
      message, 'Skip file because data are not [x,y,wave]: '+file, /continue
      CONTINUE
    ENDIF

    sicg_l2_make_annulus, sz[1], sz[2], x0, y0, rlimb, rmax, annmask
    IF keyword_set(despike) THEN sicg_l2_smooth, cube, annmask, smooth_size=smooth_size

    sicg_l2_three_point, cube, wave, bandpass, x0, y0, rlimb, rmax, inten, dop, wid
    sicg_l2_correct_doppler, dop, x0, y0, rlimb, rmax, dop_corr, dop_col_median, $
                            dop_limit=dop_limit, poly_order=poly_order

    int_out = inten
    dop_out = dop_corr
    wid_out = wid

    IF keyword_set(write_fits) THEN BEGIN
      datestr = sicg_l2_datestr_from_hdr(hdr, file)

      int_file = int_dir + 'OYULO_SICG01_PKIM_L2_STP_' + datestr + '_V01.00_' + expo + '.FITS'
      dop_file = dop_dir + 'OYULO_SICG01_ECDG_L2_STP_' + datestr + '_V01.00_' + expo + '.FITS'
      wid_file = wid_dir + 'OYULO_SICG01_CLWM_L2_STP_' + datestr + '_V01.00_' + expo + '.FITS'

      sicg_l2_write_one, int_file, inten, hdr, 'Coronal Peak-Intensity Maps', '1.E-06 B/Bsun'
      sicg_l2_write_one, dop_file, dop_corr, hdr, 'Coronal Dopplergrams', 'km/s'
      sicg_l2_write_one, wid_file, wid, hdr, 'Coronal Line-Width Maps', 'Angstrom'

      IF NOT keyword_set(quiet) THEN BEGIN
        print, 'Written: ', int_file
        print, 'Written: ', dop_file
        print, 'Written: ', wid_file
      ENDIF
    ENDIF
  ENDFOR
END
