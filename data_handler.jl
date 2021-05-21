using Match
using Base.Filesystem
using CSV
using Statistics
# using JuliaDB


function download_and_extract(set::String)

    url = @match set begin
        "destillation_column" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/destill.dat.gz"
        "glass_furnace" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/glassfurnace.dat.gz"
        "power_plant" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/powerplant.dat.gz"
        "evaporator" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/evaporator.dat.gz"
        "stirring_tank" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/pHdata.dat.gz"
        "fractional_distillation_column" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/distill2.dat.gz"
        "industrial_dryer" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/dryer2.dat.gz"
        "heat_exchanger" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/exchanger.dat.gz"
        "winding_process" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/winding.dat.gz"
        "stirred_tank_reactor" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/cstr.dat.gz"
        "power_plant_2" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/steamgen.dat.gz"
        "ball_and_beam" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/ballbeam.dat.gz"
        "hair_dryer" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/dryer.dat.gz"
        "cd_arm" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/CD_player_arm.dat.gz"
        "wing_flutter" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/flutter.dat.gz"
        "flexible_arm" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/robot_arm.dat.gz"
        "steel_subframe" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/flexible_structure.dat.gz"
        "pregnant_woman" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/biomedical/foetal_ecg.dat.gz"
        "tongue_displacement" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/biomedical/tongue.dat.gz"
        "lake_erie" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/environmental/erie.dat.gz"
        "two_layer_wall" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/thermic/thermic_res_wall.dat.gz"
        "heating_system" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/thermic/heating_system.dat.gz"
        "internet_traffic" => "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/timeseries/internet_traffic.dat.gz"
        "silverbox" => "http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip"
        "cascaded_tanks" => "http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/CASCADEDTANKS/CascadedTanksFiles.zip"
        "f16" => "http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/F16/F16GVT_Files.zip"
        "emps" => "http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/EMPS/EMPS.zip"
        "wiener_hammerstein" => "http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/WIENERHAMMERSTEINPROCESS/WienerHammersteinFiles.zip"
    end
    try
        print("Making directory ", "./data", "...")
        mkdir("./data")
        println("Done!")
    catch
        println("Directory already exists.")
    end

    # paths and file names
    set_name = split(url, "/")[end]
    folder = string("./data/", set, "/")
    file_path = string(folder, set_name)
    unzipped_file_name = string(split(set_name, ".")[1], "")
    unzipped_file_path = string(folder, unzipped_file_name)

    
    try
        print("Making directory ", folder, "...")
        mkdir(folder)
        println("Done!")
    catch
        println("Directory already exists.")
    end

    # Download data. If data exists, return
    print("Downloading data from ", url, "...")
    if isfile(string(unzipped_file_path, ".zip")) || isdir(unzipped_file_path) || isfile(string(unzipped_file_path, ".dat"))
        println("File already exists.")
        return unzipped_file_path
    end

    download(url, file_path);
    println("Done!")

    # Unzip based on file type
    file_type = split(set_name, ".")[end]
    cmd = @match file_type begin
        "gz" => `gzip -d $file_path`
        "zip" => `unzip $file_path -d $folder`
    end
    run(cmd)

    # remove the .gz to get the atual file name
    return file_path
end

"""
Loads the data, filters missing values and casts.
"""
function load_io_data(fp::String, dtype=Float64; header=false)
    
    filter_missing(row) = Iterators.filter(r -> !ismissing(r), row)
    cast(row) = map(r -> dtype(r), row)
    collate(row) = Iterators.reduce(vcat, row)
    process_row(row) = row |> filter_missing |> cast |> collate

    data = map(process_row, CSV.File(fp, header=header))
    return data
end

# data pre-processing
# returns a whitener and dewhitener
function standardize(signal)
    μ = mean(signal)
    σ = std(signal)

    process(zi) = (zi - μ) ./ σ
    return process
end

function unstandardize(signal)
    μ = mean(signal)
    σ = std(signal)

    unprocess(zi) = σ .* zi .+ μ
    return unprocess
end


function load_silverbox()
    fp = "./data/silverbox/SilverboxFiles/SNLS80mV.csv"
    data = load_io_data(fp; header=true)
    u_raw = map(d -> d[1:1, :], data)
    y_raw = map(d -> d[2:2, :], data)

    u = standardize(u_raw).(u_raw)
    y = standardize(y_raw).(y_raw)

    uv = u[1:40400]
    yv = y[1:40400]

    ut = u[40400:end]
    yt = y[40400:end]

    stats = (μu = 0.0, μy = 0.0, σu = 1.0, σy = 1.0)

    return (ut, yt), (uv, yv), stats
end

function load_cascaded_tanks()
    fp = "./data/cascaded_tanks/CascadedTanksFiles/dataBenchmark.csv"
    raw_data = load_io_data(fp; header=true)
    raw_data[1] = raw_data[1][1:4]
    # data = standardize(raw_data).(raw_data)

    ut = map(d -> d[1:1, :], raw_data)
    yt = map(d -> d[2:2, :], raw_data)
    
    μu = mean(ut)
    σu = std(ut)
    μy = mean(yt)
    σy = std(yt)

    uv = standardize(ut).(map(d -> d[3:3, :], raw_data))
    yv = standardize(yt).(map(d -> d[4:4, :], raw_data))

    ut = standardize(ut).(ut)
    yt = standardize(yt).(yt)
    
    return (ut, yt), (uv, yv), (μu = μu, μy = μy, σu = σu, σy = σy)
end

function load_wing_flutter()
    fp = "./data/wing_flutter/flutter.dat"
    data = load_io_data(fp)

    u_raw = map(d -> d[1:1, :], data)
    y_raw = map(d -> d[2:2, :], data)

    u = standardize(u_raw).(u_raw)
    y = standardize(y_raw).(y_raw)

    uv = u
    yv = y
    ut = u
    yt = y

    # Currently not bothering accounting for stats...
    stats = (μu = 0.0, μy = 0.0, σu = 1.0, σy = 1.0) 
    return (ut, yt), (uv, yv), stats
end

function load_flexible_arm()
    fp = "./data/flexible_arm/robot_arm.dat"
    data = load_io_data(fp)

    u_raw = map(d -> d[1:1, :], data)
    y_raw = map(d -> d[2:2, :], data)

    u = standardize(u_raw).(u_raw)
    y = standardize(y_raw).(y_raw)

    uv = u
    yv = y
    ut = u
    yt = y

    # Currently not bothering accounting for stats...
    stats = (μu = 0.0, μy = 0.0, σu = 1.0, σy = 1.0) 
    return (ut, yt), (uv, yv), stats
end

function load_liquid_saturated_heat_exchanger()
    fp  = "./data/heat_exchanger/exchanger.dat"
    data = load_io_data(fp; header=false)

    u_raw = map(d -> d[2:2, :], data)
    y_raw = map(d -> d[3:3, :], data)

    μu = mean(u_raw)
    σu = std(u_raw)
    μy = mean(y_raw)
    σy = std(y_raw)

    u = standardize(u_raw).(u_raw)
    y = standardize(y_raw).(y_raw)

    uv = u[1000:3000]
    yv = y[1000:3000]

    ut = u[1:1000]
    yt = y[1:1000]
    stats = (μu = μu, μy = μy, σu = σu, σy = σy)
    return (ut, yt), (uv, yv), stats

end

function load_f16()
    fp1 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level1.csv"
    fp2 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level2_Validation.csv"
    fp3 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level3.csv"
    fp4 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level4_Validation.csv"
    fp5 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level5.csv"
    fp6 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level6_Validation.csv"
    fp7 = "./data/f16/F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level7.csv"

    split_data(xi) = (xi[1:2, :], xi[3:5, :])
    row_filt(D) = map(x -> x[1:5], D)
    stand = standardize(row_filt(load_io_data(fp4, header=true)))  # use this dset for standarization

    loader(fp) = split_data.(stand.(row_filt(load_io_data(fp, header=true))))

    u1, y1 = unzip(loader(fp1))
    u2, y2 = unzip(loader(fp2))
    u3, y3 = unzip(loader(fp3))
    u4, y4 = unzip(loader(fp4))
    u5, y5 = unzip(loader(fp5))
    u6, y6 = unzip(loader(fp6))
    u7, y7 = unzip(loader(fp7))

    ut = hcat.(u1, u3, u5, u7)
    yt = hcat.(y1, y3, y5, y7)
    uv = hcat.(u2, u4, u6)
    yv = hcat.(y2, y4, y6)
   
    stats = (μu = 0.0, μy = 0.0, σu = 1.0, σy = 1.0)
    return (ut, yt), (uv, yv), stats
end


# 
# Identification is model (r, u) -> y
# 
function load_wiener_hammerstein()
    fp1 = "./data/wiener_hammerstein/WienerHammersteinFiles/WH_MultisineFadeOut.csv"
    fp2 = "./data/wiener_hammerstein/WienerHammersteinFiles/WH_TestDataset.csv"
    # fp3 = "./data/wiener_hammerstein/WienerHammersteinFiles/WH_SineSweepInput_meas.csv"

    row_filt(D) = map(x -> x[1:6], D)
    raw_train_data = row_filt(load_io_data(fp1, header=true))

    # Creater a standardizer for the inputs and outputs
    # D = [[di[1], di[2], di[5]] for di in raw_train_data]
    # stand = standardize(raw_train_data)

    # split into inputs and outputs, for two experiments
    split_data(data) = map(d -> ([d[1] d[2]; d[3] d[4]], [d[5] d[6]]), data)  
    ut, yt = unzip((split_data(raw_train_data)))

    σu = std(ut)[:, 1][:, :]
    σy = std(yt)[:, 1][:, :]
    μu = mean(ut)[:, 1][:, :]
    μy = mean(yt)[:, 1][:, :]

    S(di) = ((di[1].-μu) ./ σu, (di[2].-μy) ./ σy)

    test_data = row_filt(load_io_data(fp2, header=true))

    ut, yt = unzip(S.(split_data(raw_train_data)))
    uv, yv = unzip(S.(split_data(test_data)))


    # statistics = (σy = σ[5:6], σu = σ[1:4], μu = μ[1:4], μy=μ[5:6])
    statistics = (σy = σy, σu = σu, μu = μu, μy = μy)
    return (ut, yt), (uv, yv), statistics
end