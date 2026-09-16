import re

from freqtrade.constants import Config


_VALID_PAIR_RE = re.compile(r"[\w:/-]+")


def expand_pairlist(
    wildcardpl: list[str],
    available_pairs: list[str],
    keep_invalid: bool = False,
    *,
    use_regex: bool = True,
) -> list[str]:
    """
    Expand pairlist potentially containing wildcards based on available markets.
    This will implicitly filter all pairs in the wildcard-list which are not in available_pairs.
    :param wildcardpl: List of Pairlists, which may contain regex
    :param available_pairs: List of all available pairs (`exchange.get_markets().keys()`)
    :param keep_invalid: Retain unmatched pair names, then remove entries containing invalid
        pair-name characters or underscores. Applies to both regex and literal matching.
    :param use_regex: Interpret entries as regular expressions (default). If False, match literal
        symbols case-insensitively, preserving available market spelling and input order.
    :return: Matched pairs, preserving input duplicates and available market order per match.
    :raises: ValueError if use_regex is True and a wildcard is invalid
        (like '*/BTC' - which should be `.*/BTC`)
    """
    literal_pairs: dict[str, list[tuple[int, str]]] | None = None
    unicode_pairs: list[tuple[int, str]] = []
    if not use_regex:
        literal_pairs = {}
        for index, pair in enumerate(available_pairs):
            if pair.isascii():
                literal_pairs.setdefault(pair.lower(), []).append((index, pair))
            else:
                unicode_pairs.append((index, pair))

    result = []
    for pair_wc in wildcardpl:
        try:
            if literal_pairs is not None and pair_wc.isascii():
                matches = literal_pairs.get(pair_wc.lower(), [])
                if unicode_pairs:
                    # Unicode symbols can match ASCII input under IGNORECASE. Check only
                    # those markets and merge by position to retain available market order.
                    comp = re.compile(re.escape(pair_wc), re.IGNORECASE)
                    unicode_matches = [item for item in unicode_pairs if comp.fullmatch(item[1])]
                    if unicode_matches:
                        matches = sorted(matches + unicode_matches)
                result_partial = [pair for _, pair in matches]
            else:
                # Unicode IGNORECASE is not equivalent to lower()/casefold(). Keep its
                # matching rules for non-ASCII symbols, escaping regex syntax in literal mode.
                pattern = pair_wc if use_regex else re.escape(pair_wc)
                comp = re.compile(pattern, re.IGNORECASE)
                result_partial = [pair for pair in available_pairs if comp.fullmatch(pair)]
            # Keep unmatched input only when requested; do not deduplicate matched markets.
            result += result_partial or ([pair_wc] if keep_invalid else [])
        except re.error as err:
            raise ValueError(f"Wildcard error in {pair_wc}, {err}")

    if keep_invalid:
        # Remove wildcard pairs that didn't have a match.
        result = [
            element
            for element in result
            if _VALID_PAIR_RE.fullmatch(element) and "_" not in element
        ]

    return result


def dynamic_expand_pairlist(config: Config, markets: list[str]) -> list[str]:
    expanded_pairs = expand_pairlist(config["pairs"], markets)
    if config.get("freqai", {}).get("enabled", False):
        corr_pairlist = config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        expanded_pairs += [pair for pair in corr_pairlist if pair not in config["pairs"]]

    return expanded_pairs
