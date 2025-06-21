import idaes.logger as idaeslog
import logging

from pyomo.environ import (
    Constraint,
    Objective,
    Var,
    Expression,
    Param,
    ConcreteModel,
    TransformationFactory,
    value,
    maximize,
    SolverFactory,
    units as pyunits,
)
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import FlowsheetBlock
from idaes.core import UnitModelCostingBlock
from idaes.core.solvers import get_solver
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import InitializationError
from idaes.models.costing.SSLW import SSLWCosting

from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption

from idaes_examples.mod.hda import hda_ideal_VLE as thermo_props
from idaes_examples.mod.hda import hda_reaction as reaction_props

from idaes.models.unit_models import (
    PressureChanger,
    Mixer,
    Separator as Splitter,
    Heater,
    CSTR,
    StoichiometricReactor,
    Flash
)

def function(unit):
    try:
        initializer = unit.default_initializer()
        initializer.initialize(unit)
    except InitializationError:
        solver = get_solver()
        solver.solve(unit)

def hda_objective(H101_temperature, F101_temperature, F102_temperature, F102_deltaP, metric='cost', 
                  log=False, 
                  f101_cooling_cost=0.212e-7, 
                  h101_heating_cost=2.2e-7, f102_heating_cost=1.9e-7,
                  h101_heating_cost_perW = 63931.475, benzene_price=820,
                  hydrogen_vap_price=16.51,
                  toluene_liq_price=0.77 ):
    if log == False:
        logging.getLogger("idaes").setLevel(logging.ERROR)
        logging.getLogger("pyomo").setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)
    
    # Model initialization
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Configure property packages
    m.fs.thermo_params = thermo_props.HDAParameterBlock()
    m.fs.reaction_params = reaction_props.HDAReactionParameterBlock(
        property_package=m.fs.thermo_params
    )

    # Unit model definitions
    m.fs.M101 = Mixer(
    property_package=m.fs.thermo_params,
    inlet_list=["toluene_feed", "hydrogen_feed", "vapor_recycle"],
    )

    # Heater unit for process stream temperature adjustment
    m.fs.H101 = Heater(
    property_package=m.fs.thermo_params,
    has_pressure_change=False,
    has_phase_equilibrium=False,
    )

    m.fs.R101 = StoichiometricReactor(
            property_package=m.fs.thermo_params,
            reaction_package=m.fs.reaction_params,
            has_heat_of_reaction=True,
            has_heat_transfer=True,
            has_pressure_change=False)

    # Flash separation unit for vapor-liquid separation
    m.fs.F101 = Flash(
    property_package=m.fs.thermo_params,
    has_heat_transfer=True,
    has_pressure_change=False
    )

    # Stream splitter for purge and recycle streams
    m.fs.S101 = Splitter(
    property_package=m.fs.thermo_params,
    ideal_separation=False,
    outlet_list=["purge", "recycle"],
    )

    # Compressor for recycle stream
    m.fs.C101 = PressureChanger(
    property_package=m.fs.thermo_params,
    compressor=True,
    thermodynamic_assumption=ThermodynamicAssumption.isothermal,
    )

    # Secondary flash unit for further separation
    m.fs.F102 = Flash(
    property_package=m.fs.thermo_params,
    has_heat_transfer=True,
    has_pressure_change=True,
    )

    # Define process stream connections using Arcs
    m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.H101.inlet)

    # Connect heater outlet to reactor inlet
    m.fs.s04 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)

    # Connect reactor outlet to primary flash unit
    m.fs.s05 = Arc(source=m.fs.R101.outlet, destination=m.fs.F101.inlet)

    # Connect flash vapor outlet to splitter
    m.fs.s06 = Arc(source=m.fs.F101.vap_outlet, destination=m.fs.S101.inlet)

    # Connect splitter recycle stream to compressor
    m.fs.s08 = Arc(source=m.fs.S101.recycle, destination=m.fs.C101.inlet)

    # Complete recycle loop back to mixer
    m.fs.s09 = Arc(source=m.fs.C101.outlet, destination=m.fs.M101.vapor_recycle)

    # Connect flash liquid outlet to secondary flash
    m.fs.s10 = Arc(source=m.fs.F101.liq_outlet, destination=m.fs.F102.inlet)


    # Apply network expansion to connect all arcs in the flowsheet
    TransformationFactory("network.expand_arcs").apply_to(m)


    # Define product purity expression based on benzene to benzene+toluene ratio in vapor outlet
    m.fs.purity = Expression(
    expr=m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"]
    / (
        m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"]
        + m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "toluene"]
        + m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "methane"]
        + m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "hydrogen"]
    )
    )

    # Set feed conditions for toluene feed stream to mixer
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-10)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-10)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(1e-10)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "methane"].fix(1e-10)

    # Set liquid phase composition for toluene feed
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-10)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "toluene"].fix(0.30)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-10)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-10)

    # Fix toluene feed temperature and pressure
    m.fs.M101.toluene_feed.temperature.fix(303.2)
    m.fs.M101.toluene_feed.pressure.fix(350000)

    # Set feed conditions for hydrogen feed stream to mixer
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-10)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-10)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(0.30)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "methane"].fix(0.02)

    # Set liquid phase composition for hydrogen feed (trace amounts)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-10)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "toluene"].fix(1e-10)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-10)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-10)

    # Fix hydrogen feed temperature and pressure
    m.fs.M101.hydrogen_feed.temperature.fix(303.2)
    m.fs.M101.hydrogen_feed.pressure.fix(350000)

    # Define reactor conversion variable with initialization and bounds
    m.fs.R101.conversion = Var(initialize=0.75, bounds=(0, 1))

    m.fs.R101.conv_constraint = Constraint(
    expr=m.fs.R101.conversion*m.fs.R101.inlet.
    flow_mol_phase_comp[0, "Vap", "toluene"] ==
    (m.fs.R101.inlet.flow_mol_phase_comp[0, "Vap", "toluene"] -
     m.fs.R101.outlet.flow_mol_phase_comp[0, "Vap", "toluene"]))

    # Fix reactor conversion at 75%
    m.fs.R101.conversion.fix(0.75)

    # No pressure drop in first flash unit
    # m.fs.F101.deltaP.fix(0)

    # Set purge split fraction to control recycle rate
    m.fs.S101.split_fraction[0, "purge"].fix(0.2)

    # Set compressor outlet pressure for recycle stream
    m.fs.C101.outlet.pressure.fix(350000)

    # Set heater outlet temperature for reactor feed
    m.fs.H101.outlet.temperature.fix(H101_temperature)

    # Set reactor heat duty to adiabatic operation
    m.fs.R101.heat_duty.fix(1e-5)

    # Set flash separator operating conditions
    m.fs.F101.vap_outlet.temperature.fix(F101_temperature)

    # Set secondary flash unit temperature
    m.fs.F102.vap_outlet.temperature.fix(F102_temperature)

    # Set pressure drop in secondary flash unit
    m.fs.F102.deltaP.fix(F102_deltaP)

    # Define cooling cost expression from process heat duties
    m.fs.cooling_cost = Expression(
    expr= f101_cooling_cost * (-m.fs.F101.heat_duty[0]) + f101_cooling_cost * (-m.fs.R101.heat_duty[0])
    )

    # Define heating cost expression from process heating requirements
    m.fs.heating_cost = Expression(
    expr=h101_heating_cost * m.fs.H101.heat_duty[0] + f102_heating_cost * m.fs.F102.heat_duty[0]
    )

    m.fs.operating_cost = Expression(
    expr=(3600 * 24 * 365 * (m.fs.heating_cost + m.fs.cooling_cost))
    )

    # Computing reactor capital cost
    m.fs.R101.diameter = Param(initialize=2, units=pyunits.m)
    m.fs.R101.length = Var(initialize=6, units=pyunits.m)

    m.fs.costing = SSLWCosting()
    m.fs.R101.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    m.fs.R101.length.unfix()
    m.fs.R101.L_eq = Constraint(
    expr=m.fs.R101.length
    == 13.2000 * pyunits.m * m.fs.R101.conversion - 5.9200 * pyunits.m
    )

    # Computing flash capital cost
    m.fs.F101.diameter = Param(initialize=2, units=pyunits.m)
    m.fs.F101.length = Param(initialize=4, units=pyunits.m)
    m.fs.F101.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    m.fs.F102.diameter = Param(initialize=2, units=pyunits.m)
    m.fs.F102.length = Param(initialize=4, units=pyunits.m)
    m.fs.F102.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    # Computing heater/cooler capital costs
    m.fs.H101.cost_heater = Expression(
    expr=h101_heating_cost* m.fs.H101.heat_duty[0] + h101_heating_cost_perW * pyunits.W,
    doc="capital cost of heater in $",
    )

    # Annualizing capital cost to same scale as operating costs (per year)
    m.fs.annualized_capital_cost = Expression(
    expr=(
        m.fs.R101.costing.capital_cost
        + m.fs.F101.costing.capital_cost
        + m.fs.F102.costing.capital_cost
        + m.fs.H101.cost_heater
    )
    * 5.4
    / 15
    )

    #  C6H6 -- 12*6 + 6 = 78 g/mol
    # benzeze price = 820/ton -- https://businessanalytiq.com/procurementanalytics/index/benzene-price-index/
    # - 1 gr = 1e-6 MT  -- consider 1000
    m.fs.sales = Expression(
    expr=(
        3600 * 24 * 365 * m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] # flash 2 benzebe flow rate
        * 78
        * 1e-6
        * benzene_price
        * 1000
    )
    ) # unit USD/kg

    # H2 $16.51 per kilogram - 2.016 g/mol
    # Toluene $0.77 per kilogram - 92 g/mol
    m.fs.raw_mat_cost = Expression(
    expr=(
        3600 * 24 * 365 * m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "toluene"] * toluene_liq_price * 92 / 1000
        + 3600 * 24 * 365 * m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"] * hydrogen_vap_price * 2.016 / 1000
    )
    ) # unit USD/kg

    # Configure sequential decomposition solver for handling recycle streams
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 3

    # Using the SD tool
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    order = seq.calculation_order(G)

    # Define tear stream initial guesses for sequential solver
    tear_guesses = {
        "flow_mol_phase_comp": {
            (0, "Vap", "benzene"): 1e-10,
            (0, "Vap", "toluene"): 1e-10,
            (0, "Vap", "hydrogen"): 0.30,
            (0, "Vap", "methane"): 0.02,
            (0, "Liq", "benzene"): 1e-10,
            (0, "Liq", "toluene"): 0.30,
            (0, "Liq", "hydrogen"): 1e-10,
            (0, "Liq", "methane"): 1e-10,
        },
        "temperature": {0: 303.2},
        "pressure": {0: 350000},
    }

    # Pass the tear_guess to the SD tool
    seq.set_guesses_for(m.fs.H101.inlet, tear_guesses)

    # Run sequential decomposition algorithm to solve flowsheet with recycle
    seq.run(m, function)

    # Create the solver object
    solver = get_solver()

    # Solve the model
    results = solver.solve(m, tee=False) 

    if metric == 'cost':
        return value(m.fs.operating_cost)
    elif metric == 'yield':
        yield_benzene = value(m.fs.R101.outlet.flow_mol_phase_comp[0, "Vap", "benzene"]) * 3600 * 24 * 365
        return yield_benzene
    elif metric == 'yield/cost':
        yield_benzene = value(m.fs.R101.outlet.flow_mol_phase_comp[0, "Vap", "benzene"]) * 3600 * 24 * 365
        return yield_benzene/value(m.fs.operating_cost)


if __name__ == "__main__":

    # define cost coefficients
    f101_cooling_cost=0.212e-7
    h101_heating_cost=2.2e-7
    f102_heating_cost=1.9e-7
    h101_heating_cost_perW = 63931.475
    benzene_price=820
    hydrogen_vap_price=16.51
    toluene_liq_price=0.77

    output = hda_objective(
        H101_temperature=827.2, 
        F101_temperature=305.6, 
        F102_temperature=306.6, 
        F102_deltaP=-48444,
        metric='cost',
        log=False, 
        f101_cooling_cost=f101_cooling_cost, 
        h101_heating_cost=h101_heating_cost, 
        f102_heating_cost=f102_heating_cost,
        h101_heating_cost_perW = h101_heating_cost_perW, 
        benzene_price=benzene_price,
        hydrogen_vap_price=hydrogen_vap_price,
        toluene_liq_price=toluene_liq_price
    )
    print(output)