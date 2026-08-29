import pytest


"""

This module tests the front end: turning an input line into a Calculation object, and the command line
entry point that users actually type at.

None of these tests run an SCF cycle, so they are fast, and they cover the part of TUNA that every
single calculation goes through.

"""


# CODATA 2022, so that the conversion is not checked against the same constant TUNA uses

BOHR_RADIUS_IN_ANGSTROM = 0.529177210544




def test_input_line_is_split_into_its_sections(tuna):

    """

    Checks that each colon-separated section ends up where it should.

    """

    calculation, atomic_symbols, coordinates = tuna.calculation("SPE : H F 0.9168 : MP2 CC-PVDZ")

    assert calculation.calculation_type == "SPE"
    assert calculation.method.name == "MP2"
    assert calculation.basis == "CC-PVDZ"
    assert atomic_symbols == ["H", "F"]
    assert coordinates.shape == (2, 3)




def test_bond_length_is_converted_from_angstroms_to_bohr(tuna):

    """

    The input line is in angstroms and everything inside TUNA is in bohr.

    """

    _, _, coordinates = tuna.calculation("SPE : H H 0.74 : HF STO-3G")

    assert coordinates[0].tolist() == [0.0, 0.0, 0.0]
    assert coordinates[1][2] == pytest.approx(0.74 / BOHR_RADIUS_IN_ANGSTROM, rel = 1e-9)




def test_a_single_atom_is_recognised_as_monatomic(tuna):

    """

    Atoms have one symbol, no bond length, and switch off the molecular calculation types.

    """

    calculation, atomic_symbols, coordinates = tuna.calculation("SPE : HE : HF CC-PVDZ")

    assert atomic_symbols == ["HE"]
    assert calculation.monatomic
    assert not calculation.diatomic
    assert coordinates.shape == (1, 3)




def test_a_ghost_atom_makes_the_calculation_monatomic(tuna):

    """

    A molecule with one real atom and one ghost is treated as an atom in a bigger basis.

    """

    calculation, _, _ = tuna.calculation("SPE : H XH 0.74 : HF 6-31G")

    assert calculation.ghost_atom_present
    assert calculation.monatomic




@pytest.mark.parametrize("keyword, attribute, expected", [

    ("CH 1", "charge", 1),
    ("ML 3", "multiplicity", 3),
    ("MAXITER 250", "max_iter", 250),
    ("TEMP 500", "temperature", 500.0),
    ("NODIIS", "DIIS", False),
    ("DECONTRACT", "decontract", True),
    ("FREEZECORE", "freeze_core", True),
    ("ROOT 3", "root", 3),

    ])
def test_keywords_set_calculation_attributes(tuna, keyword, attribute, expected):

    """

    Checks that a keyword in the fourth section reaches the attribute it is meant to set.

    """

    calculation, _, _ = tuna.calculation(f"SPE : H H 0.74 : HF STO-3G : {keyword}")

    assert getattr(calculation, attribute) == expected




@pytest.mark.parametrize("first, second", [

    ("CH 1", "CHARGE 1"),
    ("ML 3", "MULTIPLICITY 3"),
    ("TEMP 500", "TEMPERATURE 500"),
    ("GEOMMAXITER 5", "MAXGEOMITER 5"),

    ])
def test_keyword_aliases_are_equivalent(tuna, first, second):

    """

    Two spellings of a keyword have to set the same attribute to the same value.

    """

    calculation_first, _, _ = tuna.calculation(f"SPE : H H 0.74 : HF STO-3G : {first}")
    calculation_second, _, _ = tuna.calculation(f"SPE : H H 0.74 : HF STO-3G : {second}")

    attribute = {"CH": "charge", "ML": "multiplicity", "TEMP": "temperature", "GEOMMAXITER": "geom_max_iter"}[first.split()[0]]

    assert getattr(calculation_first, attribute) == getattr(calculation_second, attribute)




def test_an_unrestricted_method_is_flagged_as_unrestricted(tuna):

    """

    A leading U selects an unrestricted reference for the method that follows it.

    """

    calculation, _, _ = tuna.calculation("SPE : H F 0.9168 : UHF 6-31G")

    assert calculation.method.unrestricted

    calculation, _, _ = tuna.calculation("SPE : H F 0.9168 : HF 6-31G")

    assert not calculation.method.unrestricted




@pytest.mark.parametrize("input_line, problem", [

    ("SPE : H H 0.74 : NOTAMETHOD STO-3G",  "unknown method"),
    ("SPE : H H 0.74 : HF NOTABASIS",       "unknown basis set"),
    ("SPE : XX YY 0.74 : HF STO-3G",        "unknown atoms"),
    ("NOTACALC : H H 0.74 : HF STO-3G",     "unknown calculation type"),
    ("SPE : H H : HF STO-3G",               "missing bond length"),
    ("SPE : H H 0.001 : HF STO-3G",         "bond length below the minimum"),
    ("SPE : H H 0.74 : HF",                 "malformed input line"),
    ("SPE : H H 0.74 : ULMP2 STO-3G",       "unrestricted reference not available"),

    ])
def test_bad_input_is_rejected(tuna, input_line, problem):

    """

    Every one of these should be caught by the front end and reported, not allowed through.

    """

    with pytest.raises(tuna.TunaError):

        tuna.calculation(input_line)




def test_command_line_prints_the_version(tuna):

    """

    The version flag is the check the README tells users to run after installing.

    """

    result = tuna.command_line("--version")

    assert tuna.version in result.output




def test_command_line_runs_a_single_point(tuna):

    """

    A smoke test of the path a user actually takes, which the in-process tests bypass.

    """

    result = tuna.command_line("SPE : H H 0.74 : HF STO-3G")

    assert result.returncode == 0
    assert "-1.1167593" in result.output
    assert "completed successfully" in result.output




def test_command_line_reports_an_error_and_exits_nonzero(tuna):

    """

    A bad input line should fail loudly rather than producing a traceback or a zero exit code.

    """

    result = tuna.command_line("SPE : H H 0.74 : NOTAMETHOD STO-3G")

    assert result.returncode == 1
    assert "ERROR" in result.output
    assert "NOTAMETHOD" in result.output
