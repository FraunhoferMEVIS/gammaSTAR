use ezpc::*;

use super::{helpers::*, *};

pub fn version() -> Parser<impl Parse<Output = Version>> {
    let major = tag_ws("{gradz_t}") + int() + nl();
    // let minor = tag_ws("minor") + int() + nl();
    // let revision = tag_ws("revision") + int() + ident().opt() + nl();

    ().map(
        |((major))| Version {
            major
        },
    )
}