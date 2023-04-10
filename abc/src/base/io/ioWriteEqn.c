/**CFile****************************************************************

  FileName    [ioWriteEqn.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [Command processing package.]

  Synopsis    [Procedures to write equation representation of the network.]

  Author      [Alan Mishchenko]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 1.0. Started - June 20, 2005.]

  Revision    [$Id: ioWriteEqn.c,v 1.00 2005/06/20 00:00:00 alanmi Exp $]

***********************************************************************/

#include "ioAbc.h"

ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

static void Io_NtkWriteEqnOne( FILE * pFile, Abc_Ntk_t * pNtk );
static void Io_NtkWriteEqnCis( FILE * pFile, Abc_Ntk_t * pNtk );
static void Io_NtkWriteEqnCos( FILE * pFile, Abc_Ntk_t * pNtk );

static int Io_NtkWriteEqnCheck( Abc_Ntk_t * pNtk );



//static void Io_WriteEqnTorch( Abc_Ntk_t * pNtk, char * pFileName );


static void Io_NtkWriteEqnOneTorch( FILE * pFile, Abc_Ntk_t * pNtk );
static void Io_NtkWriteEqnCosTorch( FILE * pFile, Abc_Ntk_t * pNtk );
static void Io_NtkWriteEqnCisTorch( FILE * pFile, Abc_Ntk_t * pNtk );

//
//
//
////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////
/*
void Hop_ObjPrintEqn( FILE * pFile, Hop_Obj_t * pObj, Vec_Vec_t * vLevels, int Level )
{
    Vec_Ptr_t * vSuper;
    Hop_Obj_t * pFanin;
    int fCompl, i;
    // store the complemented attribute
    fCompl = Hop_IsComplement(pObj);
    pObj = Hop_Regular(pObj);
    // constant case
    if ( Hop_ObjIsConst1(pObj) )
    {
        fprintf( pFile, "%d", !fCompl );
        return;
    }
    // PI case
    if ( Hop_ObjIsPi(pObj) )
    {
        fprintf( pFile, "%s%s", fCompl? "!" : "", (char*)pObj->pData );
        return;
    }
    // AND case
    Vec_VecExpand( vLevels, Level );
    vSuper = Vec_VecEntry(vLevels, Level);
    Hop_ObjCollectMulti( pObj, vSuper );
    fprintf( pFile, "%s", (Level==0? "" : "(") );
    Vec_PtrForEachEntry( Hop_Obj_t *, vSuper, pFanin, i )
    {
        Hop_ObjPrintEqn( pFile, Hop_NotCond(pFanin, fCompl), vLevels, Level+1 );
        if ( i < Vec_PtrSize(vSuper) - 1 )
            fprintf( pFile, " %s ", fCompl? "+" : "*" );
    }
    fprintf( pFile, "%s", (Level==0? "" : ")") );
    return;
}
*/

void Hop_ObjPrintEqnTorch( FILE * pFile, Hop_Obj_t * pObj, Vec_Vec_t * vLevels, int Level )
{
    Vec_Ptr_t * vSuper;
    Hop_Obj_t * pFanin;
    int fCompl, i;
    // store the complemented attribute
    fCompl = Hop_IsComplement(pObj);
    pObj = Hop_Regular(pObj);
    // constant case
    if ( Hop_ObjIsConst1(pObj) )
    {
        fprintf( pFile, "%d", !fCompl );
        return;
    }
    // PI case
    if ( Hop_ObjIsPi(pObj) )
    {
        fprintf( pFile, "%s%s%s", fCompl? "torch.logical_not(" : "", (char*)pObj->pData , fCompl ? ")" : "");
        return;
    }
    // AND case
    Vec_VecExpand( vLevels, Level );
    vSuper = Vec_VecEntry(vLevels, Level);
    Hop_ObjCollectMulti( pObj, vSuper );
    fprintf( pFile, "%s", (Level==0? "" : "(") );
    fprintf( pFile, " %s ", fCompl? "torch.logical_or(" : "torch.logical_and(" );
    Vec_PtrForEachEntry( Hop_Obj_t *, vSuper, pFanin, i )
    {
        Hop_ObjPrintEqnTorch( pFile, Hop_NotCond(pFanin, fCompl), vLevels, Level+1 );
        if ( i < Vec_PtrSize(vSuper) - 1 )
            fprintf( pFile, " %s ", fCompl? "," : "," );
    }
    fprintf( pFile, "%s", (Level==0? "" : ")") );
    return;
}



/**Function*************************************************************

  Synopsis    [Writes the logic network in the equation format.]

  Description []
  
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Io_WriteEqn( Abc_Ntk_t * pNtk, char * pFileName )
{
    FILE * pFile;

    assert( Abc_NtkIsAigNetlist(pNtk) );
    if ( Abc_NtkLatchNum(pNtk) > 0 )
        printf( "Warning: only combinational portion is being written.\n" );

    // check that the names are fine for the EQN format
    if ( !Io_NtkWriteEqnCheck(pNtk) )
        return;

    // start the output stream
    pFile = fopen( pFileName, "w" );
    if ( pFile == NULL )
    {
        fprintf( stdout, "Io_WriteEqn(): Cannot open the output file \"%s\".\n", pFileName );
        return;
    }
    fprintf( pFile, "# Equations for \"%s\" written by ABC on %s\n", pNtk->pName, Extra_TimeStamp() );

    // write the equations for the network
    Io_NtkWriteEqnOne( pFile, pNtk );
    fprintf( pFile, "\n" );
    fclose( pFile );
}

void Io_NtkWriteEqnOneTorch( FILE * pFile, Abc_Ntk_t * pNtk )
{
    Vec_Vec_t * vLevels;
    ProgressBar * pProgress;
    Abc_Obj_t * pNode, * pFanin;
    int i, k;

    // write the PIs
    fprintf( pFile, "import torch, numpy\n" );
    fprintf( pFile, "bool_in = torch.FloatTensor(64).uniform_() > 0.5\n" );
    fprintf( pFile, "def %s():\n", pNtk->pName );
    Io_NtkWriteEqnCisTorch( pFile, pNtk );

    // write the POs

    // write each internal node
    vLevels = Vec_VecAlloc( 10 );
    pProgress = Extra_ProgressBarStart( stdout, Abc_NtkObjNumMax(pNtk) );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        Extra_ProgressBarUpdate( pProgress, i, NULL );
        fprintf( pFile, "\t%s = ", Abc_ObjName(Abc_ObjFanout0(pNode)) );
        // set the input names
        Abc_ObjForEachFanin( pNode, pFanin, k )
            Hop_IthVar((Hop_Man_t *)pNtk->pManFunc, k)->pData = Abc_ObjName(pFanin);
        // write the formula
        Hop_ObjPrintEqnTorch( pFile, (Hop_Obj_t *)pNode->pData, vLevels, 0 );
        fprintf( pFile, ")\n" );
    }
    Extra_ProgressBarStop( pProgress );
    Vec_VecFree( vLevels );
}


void Io_WriteEqnTorch( Abc_Ntk_t * pNtk, char * pFileName )
{
    FILE * pFile;

    assert( Abc_NtkIsAigNetlist(pNtk) );
    if ( Abc_NtkLatchNum(pNtk) > 0 )
        printf( "Warning: only combinational portion is being written.\n" );

    // check that the names are fine for the EQN format
    if ( !Io_NtkWriteEqnCheck(pNtk) )
        return;

    // start the output stream
    pFile = fopen( pFileName, "w" );
    if ( pFile == NULL )
    {
        fprintf( stdout, "Io_WriteEqn(): Cannot open the output file \"%s\".\n", pFileName );
        return;
    }
    fprintf( pFile, "# Equations for \"%s\" written by ABC on %s\n", pNtk->pName, Extra_TimeStamp() );

    // write the equations for the network
    Io_NtkWriteEqnOneTorch( pFile, pNtk );
    Io_NtkWriteEqnCosTorch(pFile, pNtk);
    fprintf( pFile, "\n" );
    fclose( pFile );
}


/**Function*************************************************************

  Synopsis    [Write one network.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Io_NtkWriteEqnOne( FILE * pFile, Abc_Ntk_t * pNtk )
{
    Vec_Vec_t * vLevels;
    ProgressBar * pProgress;
    Abc_Obj_t * pNode, * pFanin;
    int i, k;

    // write the PIs
    fprintf( pFile, "INORDER =" );
    Io_NtkWriteEqnCis( pFile, pNtk );
    fprintf( pFile, ";\n" );

    // write the POs
    fprintf( pFile, "OUTORDER =" );
    Io_NtkWriteEqnCos( pFile, pNtk );
    fprintf( pFile, ";\n" );

    // write each internal node
    vLevels = Vec_VecAlloc( 10 );
    pProgress = Extra_ProgressBarStart( stdout, Abc_NtkObjNumMax(pNtk) );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        Extra_ProgressBarUpdate( pProgress, i, NULL );
        fprintf( pFile, "%s = ", Abc_ObjName(Abc_ObjFanout0(pNode)) );
        // set the input names
        Abc_ObjForEachFanin( pNode, pFanin, k )
            Hop_IthVar((Hop_Man_t *)pNtk->pManFunc, k)->pData = Abc_ObjName(pFanin);
        // write the formula
        Hop_ObjPrintEqn( pFile, (Hop_Obj_t *)pNode->pData, vLevels, 0 );
        fprintf( pFile, ";\n" );
    }
    Extra_ProgressBarStop( pProgress );
    Vec_VecFree( vLevels );
}


/**Function*************************************************************

  Synopsis    [Writes the primary input list.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Io_NtkWriteEqnCis( FILE * pFile, Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pTerm, * pNet;
    int LineLength;
    int AddedLength;
    int NameCounter;
    int i;

    LineLength  = 9;
    NameCounter = 0;

    Abc_NtkForEachCi( pNtk, pTerm, i )
    {
        pNet = Abc_ObjFanout0(pTerm);
        // get the line length after this name is written
        AddedLength = strlen(Abc_ObjName(pNet)) + 1;
        if ( NameCounter && LineLength + AddedLength + 3 > IO_WRITE_LINE_LENGTH )
        { // write the line extender
            fprintf( pFile, " \n" );
            // reset the line length
            LineLength  = 0;
            NameCounter = 0;
        }
        fprintf( pFile, " %s", Abc_ObjName(pNet) );
        LineLength += AddedLength;
        NameCounter++;
    }
}

void Io_NtkWriteEqnCisTorch( FILE * pFile, Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pTerm, * pNet;
    int LineLength;
    int AddedLength;
    int NameCounter;
    int i;

    LineLength  = 9;
    NameCounter = 0;

    Abc_NtkForEachCi( pNtk, pTerm, i )
    {
        pNet = Abc_ObjFanout0(pTerm);
        // get the line length after this name is written
        AddedLength = strlen(Abc_ObjName(pNet)) + 1;
        if ( NameCounter && LineLength + AddedLength + 3 > IO_WRITE_LINE_LENGTH )
        { // write the line extender
            fprintf( pFile, " \n" );
            // reset the line length
            LineLength  = 0;
            NameCounter = 0;
        }
        fprintf( pFile, "\t%s = bool_in\n", Abc_ObjName(pNet) );
        LineLength += AddedLength;
        NameCounter++;
    }
}


/**Function*************************************************************

  Synopsis    [Writes the primary input list.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Io_NtkWriteEqnCos( FILE * pFile, Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pTerm, * pNet;
    int LineLength;
    int AddedLength;
    int NameCounter;
    int i;

    LineLength  = 10;
    NameCounter = 0;

    Abc_NtkForEachCo( pNtk, pTerm, i )
    {
        pNet = Abc_ObjFanin0(pTerm);
        // get the line length after this name is written
        AddedLength = strlen(Abc_ObjName(pNet)) + 1;
        if ( NameCounter && LineLength + AddedLength + 3 > IO_WRITE_LINE_LENGTH )
        { // write the line extender
            fprintf( pFile, " \n" );
            // reset the line length
            LineLength  = 0;
            NameCounter = 0;
        }
        fprintf( pFile, " %s", Abc_ObjName(pNet) );
        LineLength += AddedLength;
        NameCounter++;
    }
}

void Io_NtkWriteEqnCosTorch( FILE * pFile, Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pTerm, * pNet;
    int LineLength;
    int AddedLength;
    int NameCounter;
    int i;

    LineLength  = 10;
    NameCounter = 0;
    fprintf(pFile, "\treturn ");
    Abc_NtkForEachCo( pNtk, pTerm, i )
    {
        pNet = Abc_ObjFanin0(pTerm);
        // get the line length after this name is written
        AddedLength = strlen(Abc_ObjName(pNet)) + 1;
        if ( NameCounter && LineLength + AddedLength + 3 > IO_WRITE_LINE_LENGTH )
        { // write the line extender
            //fprintf( pFile, " \n" );
            // reset the line length
            LineLength  = 0;
            NameCounter = 0;
        }
        fprintf( pFile, "%s,", Abc_ObjName(pNet) );
        LineLength += AddedLength;
        NameCounter++;
    }
    fprintf(pFile, "");
}


/**Function*************************************************************

  Synopsis    [Make sure the network does not have offending names.]

  Description []
  
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Io_NtkWriteEqnCheck( Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pObj;
    char * pName = NULL;
    int i, k, Length;
    int RetValue = 1;

    // make sure the network does not have proper names, such as "0" or "1" or containing parentheses
    Abc_NtkForEachObj( pNtk, pObj, i )
    {
        pName = Nm_ManFindNameById(pNtk->pManName, i);
        if ( pName == NULL )
            continue;
        Length = strlen(pName);
        if ( pName[0] == '0' || pName[0] == '1' )
        {
            RetValue = 0;
            break;
        }
        for ( k = 0; k < Length; k++ )
            if ( pName[k] == '(' || pName[k] == ')' || pName[k] == '!' || pName[k] == '*' || pName[k] == '+' )
            {
                RetValue = 0;
                break;
            }
        if ( k < Length )
            break;
    }
    if ( RetValue == 0 )
    {
        printf( "The network cannot be written in the EQN format because object %d has name \"%s\".\n", i, pName );
        printf( "Consider renaming the objects using command \"short_names\" and trying again.\n" );
    }
    return RetValue;
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

