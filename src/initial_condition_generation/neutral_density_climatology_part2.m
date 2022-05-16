addpath('./eos80_legacy_gamma_n/')
addpath('./eos80_legacy_gamma_n/library')

ds = load('../../data/interim/stp.mat');
size(ds.lat);
size(ds.lon);
size(ds.p);
size(ds.t_an);
size(ds.s_an);


nlat = size(ds.lat, 2)
nlon = size(ds.lon, 2)
ndepth = size(ds.p, 1)


gamma_n = zeros(ndepth, nlat, nlon);

for ilat = 1:nlat
    lat = ds.lat(ilat);
    p = ds.p(:, ilat);
    for ilon = 1:nlon
        long = ds.lon(ilon);
        if long < 0
            long = long + 360;
        end
        
        SP = ds.s_an(1, :, ilat, ilon)';
        t = ds.t_an(1, :, ilat, ilon)';
        
        place = ilat * (nlat - 1) + ilon
        if place ~= [54, 67, 72, 89, 102]  % For some reason these locations throw up errors
            [gamma_n(:, ilat, ilon), a, b] = eos80_legacy_gamma_n(SP,t,p,long,lat);
        else
            gamma_n(:, ilat, ilon) = nan;
        end
    end
end

save('../../data/interim/gamma_n.mat', 'gamma_n')