import logging
import sys
import time
import glob
import yaml
import os
import datetime
from pathlib import Path
import multiprocessing
import pickle
from multiprocessing import Pool
from itertools import count
import inspect
from pathlib import Path
os.environ['QT_PLUGIN_PATH'] = str(Path(sys.executable).parents[0]) + '\\Library\\plugins'
from helpers.plotting import Plotting
from config import Config

try:
    from calibrationlib import *
except:
    # sys.path.append("../Calibration")
    mp = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append(str(Path(mp).parents[1]) + "\\Calibration")
    sys.path.append(str(Path(mp).parents[1]))
    sys.path.append(str(Path(mp).parents[2]) + "\\Calibration")
    sys.path.append(str(Path(mp).parents[2]))
    sys.path.append(str(Path(mp).parents[3]) + "\\Calibration")
    sys.path.append(str(Path(mp).parents[3]))
    from calibrationlib import *

class AndersenMarkov:
    def __init__(self):
        print("Andersen Markov Model")

        self.model = None
        self.calibrate()


    def calibrate(self, input_folder='', working_folder='', config_path=None, base_date='', output_file='', crefile='',
                  md_file='', n_cores='', injection=''):
        if not config_path:
            config = Config()
            config_path = config.get_filepath("/data_importer/AndersenMarkovModel", "config-wti.yaml")
            # config_path = 'config-wti.yaml'
            working_folder = config.get_path_caches(dir="caches2")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = dict(yaml.load(f, Loader=yaml.FullLoader).items())
                if injection:
                    exec(injection)
        except Exception as e:
            print('Error: cannot read config file {}.'.format(config_path))
            raise Exception('Failed to read config file {}.'.format(config_path)) from e

        if not working_folder:
            working_folder = config['writableDir']

        try:
            intermediate_folder = config['intermediateFolder'].format(working_folder)
        except Exception as e:
            print('Error: cannot find intermediate folder name in config file {}.'.format(config_path))
            raise Exception('Failed to read config file {}.'.format(config_path)) from e

        try:
            os.makedirs(intermediate_folder, exist_ok=True)
        except Exception as e:
            print('Error: cannot create intermediate folder {}.'.format(intermediate_folder))
            raise Exception('Failed to create intermediate folder {}.'.format(intermediate_folder)) from e

        log_path = 'calibrate_{0}_{1}.log'.format(base_date, datetime.date.today().strftime('%d%b%Y'))
        log_path = os.path.join(intermediate_folder, log_path)
        try:
            logging.basicConfig(filename=log_path, level=logging.INFO)
        except Exception as e:
            print('Error: cannot open log file {}.'.format(log_path))
            raise Exception('Failed to open log file {}.'.format(log_path)) from e

        start_time = datetime.datetime.now()
        logging.info('Market data calibration for Adaptiv Analytics started {:%d%b%Y %H:%M}, config file {}'.\
            format(start_time, config_path))

        try:
            if not input_folder:
                input_folder = config['dataSource0']

            if not base_date:
                base_date = config['baseDate']

            if not output_file:
                output_file = config['outputFile']
            else:
                output_file = os.path.join(working_folder, output_file)

            if not n_cores:
                n_cores = multiprocessing.cpu_count() - 1
            else:
                n_cores = int(n_cores)

            curve_list = [x for x in config['majorCurrencies']]
            curve_list = [(x['calibrationCurve'], x['initialCurve']) for x in curve_list]
        except:
            logging.exception('Error reading config file {}.'.format(config_path))
            sys.exit(1)

        try:
            harv = Harvester(config, base_date, intermediate_folder, input_folder)

            try:
                harv.get('EUR DISCOUNT CURVE')
            except:
                pass
            try:
                harv.get('EUR/USD')
            except:
                pass
            try:
                harv.get('InterestRateVol/EUR')
            except:
                pass
            try:
                harv.get('InterestYieldVol/EUR')
            except:
                pass
            try:
                harv.get('FXVol/AUD/JPY')
            except:
                pass
            try:
                harv.get('WTI NYMEX')
            except:
                pass
            try:
                with open(intermediate_folder + '\\dates_monitor_{0}.csv'.format(str(base_date)), 'w') as file:
                    for i in sorted(harv.cache):
                        if harv.cache[i] is None:
                            continue
                        for j in sorted(harv.cache[i]):
                            if harv.cache[i][j] is None:
                                continue
                            if not isinstance(harv.cache[i][j][list(harv.cache[i][j])[0]], dict):
                                k = {'all': harv.cache[i][j]}
                            else:
                                k = harv.cache[i][j]
                            for l in sorted(k):
                                file.write(str(len(k[l])) if k[l] is not None else '0')
                                file.write(',' + str(i) + ' | ' + str(j) + ' | ' + str(l))
                                if k[l] is None:
                                    continue
                                for m in sorted(k[l]):
                                    file.write(',' + str(m))
                                file.write('\n')
            except:
                pass

            register = None
            ms = None

            if config['keepCache'] == 'True':
                try:
                    reg_file_name = os.path.join(intermediate_folder, base_date + '_register.f')
                    with open(reg_file_name, 'rb') as f:
                        register = pickle.load(f)
                        logging.info('Took register from {}'.format(reg_file_name))
                    model_selector_name = os.path.join(intermediate_folder, base_date + '_model_selector.f')
                    with open(model_selector_name, 'rb') as f:
                        ms = pickle.load(f)
                        logging.info('Took model selector from {}'.format(model_selector_name))

                    b_store = False
                except:
                    b_store = False
            else:
                b_store = False

            if register is None or ms is None:
                ms = ModelSelector()
                ms.s_ir = config['interestRateCalibrationAlgorithm']
                ms.s_vol = config['volCalibrationAlgorithm']
                ms.s_fx = config['fxRateCalibrationAlgorithm']
                register = ObjectRegister()

            for i in config['samples']:
                ms.add_sample(i['name'], i['offset'], i['calendar'])

            for i, j in curve_list:
                [cur, ten] = Misc.get_curve_params(i)
                [ir, newly_made, _] = register.create_instance('InterestRate', cur, ten, i)
                if newly_made:
                    ir.set_model(ms.get_model('HullWhite2FactorInterestRateModel', info=config['corrHealType'] == 'Adjusted_cov'))
                    ir.set_conservative('HullWhite2FactorInterestRateModel', register, ms, True)
                    ir.set_data(harv)
                [cur, ten] = Misc.get_curve_params(j)
                [ir, newly_made, _] = register.create_instance('InterestRate', cur, ten, j)
                if newly_made:
                    ir.set_model(False)
                    ir.set_data(harv)

            curve_list = [x for x in config['spreadCurves']]
            curve_list = [[x['baseCurve'], x['derivedCurve'], x['spreadCurve'], x['currency']] for x in curve_list]
            for j in curve_list:
                [cur, ten] = Misc.get_curve_params(j[0])
                [ir, newly_made, _] = register.create_instance('InterestRate', cur, ten, j[0])
                if newly_made:
                    ir.set_data(harv)
                [cur, ten] = Misc.get_curve_params(j[1])
                [ir1, newly_made, _] = register.create_instance('InterestRate', cur, ten, j[1])
                if newly_made:
                    ir1.set_data(harv)
                [sp, newly_made, _] = register.create_instance('InterestSpread', ir, ir1, j[2], j[3])
                if newly_made:
                    sp.set_data(harv)

            curve_list = [x for x in config['otherCurrencies']]
            curve_list = [[x['name'], x['initialCurve']] for x in curve_list]
            for i in curve_list:
                [cur, ten] = Misc.get_curve_params(i[1])
                [ir, newly_made, _] = register.create_instance('InterestRate', i[0], ten, i[1])
                if newly_made:
                    ir.set_data(harv)

            # register.for_all('set_pseudo', 'InterestRate', 'HullWhite2FactorInterestRateModel', ms)
            register.for_all('set_conservative', 'InterestRate', 'HullWhite2FactorInterestRateModel', register, ms)

            principal_curr = config['fxPrincipalCurrencies']
            mc = config['fxAdditionalFactors']

            cur_order = [x.replace('.EUR', '') for x in config['fxRates']]
            register.generate_fx(cur_order, 'EUR', principal_curr, harv, 'GBM' + str(mc), ms)

            setters = []
            targets = []
            b_delay = False
            min_hist_length = int(config['minHistoryLength'] * 7 / 5)
            check_date = (harv.last - datetime.timedelta(days=min_hist_length))
            vol_curve_list_struct = [x for x in config['impliedVolProducts']]
            for i in vol_curve_list_struct:
                for j in i['currencies'] if 'currencies' in i else i['currencyPairs']:
                    if 'currency' in j:
                        vol_type = 'InterestRateVol' if 'Rate' in j['model'] else 'InterestYieldVol'
                        [irv, newly_made, tgt] = register.create_instance('InterestRateVol', j['currency'], vol_type, vol_type + '/' + j['currency'])
                        if not newly_made:
                            continue
                        irv.set_model(ms.get_model(j['model'], j['currency']))
                        if b_delay:
                            irv.set_data(harv, True, check_date=check_date)
                            setters.append([None, irv])
                            targets.append(tgt)
                        else:
                            irv.set_data(harv, check_date=check_date)
                    elif 'currencyPair' in j:
                        vol_type = 'FXVol'
                        fx_name = j['currencyPair'].replace('.', '/')
                        if fx_name.split('/')[0] > fx_name.split('/')[1]:
                            fx_name = fx_name.split('/')[1] + '/' + fx_name.split('/')[0]
                        [fxv, newly_made, tgt] = register.create_instance('FxRateVol', fx_name, vol_type + '/' + fx_name)
                        if not newly_made:
                            continue
                        fxv.set_model(ms.get_model(j['model'], fx_name))
                        if b_delay:
                            fxv.set_data(harv, True, check_date=check_date)
                            setters.append([None, fxv])
                            targets.append(tgt)
                        else:
                            fxv.set_data(harv, check_date=check_date)

            if b_delay and len(setters) > 0:
                with Pool(n_cores) as p:
                    res = p.map(Misc.cal, setters)

                for i, j in zip(res, targets):
                    register.general_register[j] = i[1]

            for j in config['impliedVols']:
                for vol_type in ['InterestRateVol', 'InterestYieldVol']:
                    [irv, newly_made, _] = register.create_instance('InterestRateVol', j, vol_type, vol_type + '/' + j)
                    if not newly_made:
                        continue
                    irv.set_data(harv, b_last=True)

            for j in config['impliedFXVols']:
                vol_type = 'FXVol'
                fx_name = j.replace('.', '/')
                if fx_name.split('/')[0] > fx_name.split('/')[1]:
                    fx_name = fx_name.split('/')[1] + '/' + fx_name.split('/')[0]

                [fxv, newly_made, _] = register.create_instance('FxRateVol', fx_name,
                                                             vol_type + '/' + fx_name)
                if not newly_made:
                    continue
                fxv.set_data(harv)

            for j in config['majorCommodities']:
                if j['type'] == 'energy':
                    [com, newly_made, _] = register.create_instance('Energy', 'USD', '', j['name'], j['calibrationCurve'])
                    if not newly_made:
                        continue
                    com.set_model(ms.get_model('AndersenMarkov'))
                    print("energy")
                elif j['type'] == 'seasonal_energy':
                    [com, newly_made, _] = register.create_instance('Energy', 'USD', '', j['name'], j['calibrationCurve'])
                    if not newly_made:
                        continue
                    com.set_model(ms.get_model('AndersenMarkov'))
                    com.seasonal = True
                    print("seasonal_energy")
                else:
                    [com, newly_made, _] = register.create_instance('EnergySpread', 'USD', '', j['name'],
                                                                    j['spread2'], j['calibrationCurve'])
                    if not newly_made:
                        continue
                    com.set_model(ms.get_model('OrnsteinUhlenbeck'))

                # harv.get('WTI NYMEX')
                # print(harv)

                com.set_data(harv)

                # test = com.make_data()
                # print(test)


            if b_store:
                reg_file_name = os.path.join(intermediate_folder, base_date + '_register.f')
                try:
                    with open(reg_file_name, 'wb') as f:
                        pickle.dump(register, f)
                        logging.info('Saved register to {}'.format(reg_file_name))
                except:
                    logging.exception('Failed to save register to {}'.format(reg_file_name))

                model_selector_name = os.path.join(intermediate_folder, base_date + '_model_selector.f')
                try:
                    with open(model_selector_name, 'wb') as f:
                        pickle.dump(ms, f)
                        logging.info('Saved model_selector to {}.'.format(model_selector_name))
                except:
                    logging.exception('Failed to save model_selector to {}'.format(model_selector_name))

            del harv

            register.calibrate_all(n_cores, ['InterestSpread'])

            data_string = '<System Parameters>\nBase_Currency=EUR\nBase_Date={0}\n' \
                          'Correlations_Healing_Method=Eigenvalue_Raising' \
                          '\nDescription=\nExclude_Deals_With_Missing_Market_Data=Yes\nGrouping_File=\nProxying_Rules_File=\n' \
                          'Script_Base_Scenario_Multiplier=1\n\n'.format(base_date)

            data_string += '<Model Configuration>\n'
            data_string += ms.extract('Adaptiv')

            data_string += '<Price Factors>\n'
            data_string += register.extract_all('Adaptiv', 'Price Factors')
            data_string += ms.extract('AdaptivExtra')

            # Some USD trades are not discounted. Add an interest rate factor for them.
            data_string += ('InterestRate.Undiscounted USD,Property_Aliases=,Curve=[( 0,0),(100,0)],'
                'Currency=USD,Day_Count=ACT_365,Accrual_Calendar=,Sub_Type=,Floor=<undefined>\n')

            if crefile:
                with open(crefile, 'r') as file:
                    to_mod = file.read()
                try:
                    with open(output_file if not md_file else md_file, 'w') as file:
                        file.write(data_string + to_mod[to_mod.find('<Price Models>\n'):-1])
                except:
                    logging.exception('Failed to write to md mod file')

            if md_file or not crefile:
                register.calibrate_all(n_cores)

                data_string += '<Price Models>\n'
                data_string += register.extract_all('Adaptiv', 'Price Models') + '\n'

                data_string += '<Correlations>\n'
                data_string_tmp = register.extract_all('Adaptiv', 'Correlations') + '\n'
                corr = Correlator()
                corr.add(data_string_tmp)
                if config['corrHealType'] in ['Adjusted_cov', 'Yes']:
                    corr.heal()
                data_string += corr.extract('Adaptiv') + '\n'

                if 'valuationConfiguration' in config:
                    data_string += '<Valuation Configuration>\n'
                    data_string += config['valuationConfiguration'] + '\n'
                if 'marketPrices' in config:
                    data_string += '<Market Prices>\n'
                    data_string += config['marketPrices'] + '\n'
                if 'bootstrapperConfiguration' in config:
                    data_string += '<Bootstrapper Configuration>\n'
                    data_string += config['bootstrapperConfiguration'] + '\n'



                try:
                    with open(output_file, 'w') as file:
                        file.write(data_string)
                except:
                    print("logging exception")
                    logging.exception('Failed to write to {}.'.format(output_file))
                    sys.exit(1)

            # register.general_register[-1].model.make_data()

            self.model = com.model



        except:
            logging.exception('Calibration failed.')
            sys.exit(1)

        end_time = datetime.datetime.now()
        logging.info('Finished at {}, elapsed time {}.'.format(end_time, end_time - start_time))
        logging.shutdown()

    def simulate(self):
        plotting = Plotting()
        old_rates = self.model.rates

        plotting.plot_3d("AMModel_input_data", old_rates)
        plotting.plot_2d( old_rates[-1, :], "AMModel_input_data_first")

        tenors = self.model.tenors
        obs_time = self.model.obs_time

        print("tenors", tenors)
        print("obs_time", obs_time)
        print("old_rates", old_rates)

        self.model.make_data()

        rates = self.model.rates

        print("new rates", rates)

        plotting.plot_3d("AMModel_test", rates)  # , maturities=tenors, time=obs_time
        print("made data")
    # def plot_data(tenors, rates, obs_time):



if __name__ == '__main__':
    # kwargs = dict(zip(calibrate.__code__.co_varnames, sys.argv[1:]))
    # calibrate(**kwargs)
    andersenMarkov = AndersenMarkov()
    andersenMarkov.simulate()
    andersenMarkov.simulate()
    andersenMarkov.simulate()
