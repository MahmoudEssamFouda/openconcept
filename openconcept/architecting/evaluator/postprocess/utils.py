import re


def get_snopt_exit(file):
    """Scrape the SNOPT exit and info code from summary or print file.

    Parameters
    ----------
    file : str
        Summary or print file name

    Returns
    -------
    int
        SNOPT exit code
    int
        SNOPT info code
    """
    snopt_exit_pattern = re.compile("SNOPTC EXIT +[\d]* --")
    snopt_info_pattern = re.compile("SNOPTC INFO +[\d]* --")

    exit_code = None
    info_code = None

    with open(file, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            exit_match = snopt_exit_pattern.search(line)
            info_match = snopt_info_pattern.search(line)

            if exit_match:
                exit_code = int(exit_match.group().split(" ")[-2])
            if info_match:
                info_code = int(info_match.group().split(" ")[-2])

    return exit_code, info_code
